import os
import sqlite3
import operator
import yaml
import readline
import sys
from typing import Annotated, TypedDict, Sequence, List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.theme import Theme

# Настройка истории readline (чтобы стрелочки вверх/вниз работали как в баше)
readline.parse_and_bind('tab: complete')
readline.set_history_length(100)

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. ЗАГРУЗКА ПРОМПТОВ ИЗ YAML
# ==========================================
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

# ==========================================
# 2. НАСТРОЙКА ИНТЕРФЕЙСА (RICH)
# ==========================================
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "master": "bold green",
    "system": "bold yellow"
})
console = Console(theme=custom_theme)

# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ (SQLITE)
# ==========================================
DB_FILE = "ttrpg_database.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS npcs (name TEXT PRIMARY KEY, lore_and_stats TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS locations (name TEXT PRIMARY KEY, description TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS summaries (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)')
    conn.commit()
    conn.close()

init_db()

def get_all(table_name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    conn.close()
    return rows

def upsert_record(table_name, name, content):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if table_name == "npcs":
        cursor.execute("INSERT INTO npcs (name, lore_and_stats) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET lore_and_stats=excluded.lore_and_stats", (name, content))
    elif table_name == "locations":
        cursor.execute("INSERT INTO locations (name, description) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET description=excluded.description", (name, content))
    conn.commit()
    conn.close()

def save_summary(content):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO summaries (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()

# ==========================================
# 4. НАСТРОЙКА LLM
# ==========================================
llm = ChatOpenAI(
    base_url="http://192.168.2.107:5001/v1",
    api_key="local-key",
    model="Gemma-4-MeroMero",
    temperature=0.7,
    max_tokens=2000
)

# ==========================================
# 5. СКИЛЛ (ТОЛЬКО PYTHON)
# ==========================================
@tool
def python_interpreter(code: str) -> str:
    """Выполняет Python код. Используй для математики и бросков кубиков."""
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(code, {"__builtins__": __builtins__}, {})
        output = redirected_output.getvalue()
        return output if output else "Код выполнен, но ничего не выведено."
    except Exception as e:
        return f"Ошибка выполнения: {e}"
    finally:
        sys.stdout = old_stdout

tools = [python_interpreter]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 5. СТЕЙТ ГРАФА
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    npc_context: str
    is_session_end: bool
    active_agents: List[str]  # Список агентов, которых разбудит Оркестратор

# ==========================================
# 6. УЗЛЫ (АГЕНТЫ)
# ==========================================

def agent_orchestrator(state: AgentState):
    """ИИ-Оркестратор: распределяет задачи на основе ролей агентов."""
    if state.get("is_session_end"):
        return {"active_agents": ["summarizer"]}
        
    response = llm.invoke([SystemMessage(content=PROMPTS['orchestrator']), state["messages"][-1]])
    decision = response.content.lower()
    
    active = [name for name in ["npc"] if name in decision]
    if not active and not state.get("is_session_end"):
        active = []

    return {
        "active_agents": active,
        "npc_context": ""
    }

def agent_npc(state: AgentState):
    db_text = "\n".join([f"[{row[0]}]: {row[1]}" for row in get_all("npcs")])
    prompt = PROMPTS['npc'].format(db_text=db_text)
    response = llm.invoke([SystemMessage(content=prompt)] + state["messages"][-2:])
    content = response.content
    if "SAVE_NPC:" in content:
        try:
            data = content.split("SAVE_NPC:")[1].split("|", 1)
            upsert_record("npcs", data[0].strip(), data[1].strip())
        except Exception as e:
            print(f"Ошибка сохранения NPC: {e}")
    return {"npc_context": content}

def agent_master(state: AgentState):
    full_context = (
        f"--- БИБЛИОТЕКА NPC ---\n{state.get('npc_context', 'Нет данных')}"
    )
    prompt = PROMPTS['master'].format(context=full_context)
    response = llm_with_tools.invoke([SystemMessage(content=prompt)] + state["messages"])
    return {"messages": [response]}

def agent_summarizer(state: AgentState):
    response = llm.invoke([SystemMessage(content=PROMPTS['summarizer'])] + state["messages"])
    save_summary(response.content)
    return {"messages": [AIMessage(content=f"\n*** ИТОГИ СЕССИИ ***\n{response.content}")]}

# Роутеры
def route_from_orchestrator(state: AgentState):
    """Маршрутизатор из оркестратора. Запускает нужные узлы ПАРАЛЛЕЛЬНО."""
    agents = state.get("active_agents", [])
    if state.get("is_session_end"):
        return ["summarizer"]
    if not agents:
        return ["master"] # Если никто не нужен, идем напрямую к Мастеру
    return agents # Возврат списка узлов запускает их в параллельных потоках

def should_continue_from_master(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ==========================================
# 7. СБОРКА ГРАФА
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", agent_orchestrator)
workflow.add_node("npc", agent_npc)
workflow.add_node("master", agent_master)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("summarizer", agent_summarizer)

workflow.set_entry_point("orchestrator")

# МАРШРУТИЗАЦИЯ ИЗ ОРКЕСТРАТОРА
workflow.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    ["npc", "master", "summarizer"]
)

# СХОЖДЕНИЕ (Fan-in): Все параллельные агенты после работы сливаются в Мастера
workflow.add_edge("npc", "master")

# ЛОГИКА МАСТЕРА: Либо конец хода, либо вызов Python (кубики)
workflow.add_conditional_edges(
    "master",
    should_continue_from_master,
    ["tools", END]
)

# Возврат из инструментов снова к Мастеру (чтобы он озвучил результат броска)
workflow.add_edge("tools", "master")
workflow.add_edge("summarizer", END)

app = workflow.compile()

# ==========================================
# 8. UI И ГЛАВНЫЙ ЦИКЛ (STABLE INPUT MODE)
# ==========================================
def stable_input(prompt_str: str) -> str:
    """Чистый ввод через readline - не глючит как Prompt.ask в некоторых терминалах."""
    console.print(prompt_str, end="")
    try:
        return input().strip()
    except EOFError:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        raise KeyboardInterrupt

def main():
    console.clear()
    console.print(Panel.fit("🐉 [bold green]TTRPG AI MASTER[/bold green] (Stable Input Mode)", border_style="yellow"))
    
    past_history = ""
    summaries = get_all("summaries")
    if summaries:
        past_history = "История прошлых сессий:\n" + "\n".join([f"- Сессия {r[0]}: {r[1]}" for r in summaries])
        console.print("[info]База подтянута. Мастер помнит прошлые игры.[/info]")

    state: AgentState = {
        "messages": [SystemMessage(content=past_history)] if past_history else [],
        "npc_context": "",
        "is_session_end": False, "active_agents": []
    }

    while True:
        try:
            user_input = stable_input("\n[bold cyan]Игрок[/bold cyan][white] > [/white]")
        except KeyboardInterrupt:
            console.print("\n[bold red]Выход.[/bold red]")
            break
            
        if not user_input:
            continue
            
        if user_input == "/quit":
            console.print("[bold red]Выход.[/bold red]")
            break
            
        if user_input == "/end":
            state["is_session_end"] = True
            user_input = "Подведи итоги сессии."
            
        state["messages"].append(HumanMessage(content=user_input))

        # Простой последовательный вывод БЕЗ спиннера - чтобы не конфликтовал с вводом
        console.print("[dim]Оркестратор работает...[/dim]")

        try:
            for output in app.stream(state, config={"recursion_limit": 50}):
                for node_name, node_state in output.items():
                    if node_name == "orchestrator":
                        agents = node_state.get("active_agents", [])
                        if agents:
                            console.print(f"[dim grey]└─ вызваны: {', '.join(agents)}[/dim grey]")
                    
                    elif node_name == "npc":
                        console.print(f"[dim yellow]✓ {node_name.capitalize()} готов[/dim yellow]")
                        
                    elif node_name == "tools":
                        tool_msg = node_state['messages'][-1].content
                        console.print(f"[bold red]⚙️ Python:[/bold red] [white]{tool_msg.strip()}[/white]")
                        
                    elif node_name == "master":
                        console.print(f"[dim green]· Мастер пишет...[/dim green]")

            last_node = list(output.keys())[-1]
            last_msg = output[last_node]["messages"][-1]
            state["messages"].append(last_msg)
            
        except Exception as e:
            console.print(f"[bold red]Критическая ошибка:[/bold red] {e}")
            continue

        # Вывод ответа - панель отрисуется целиком и не помешает следующему вводу
        if state["is_session_end"]:
            console.print("\n")
            console.print(Panel(Markdown(last_msg.content), title="[bold gold1]ИТОГИ СЕССИИ[/bold gold1]", border_style="gold1"))
            break
        else:
            console.print("\n")
            console.print(Panel(Markdown(last_msg.content), title="[bold green]МАСТЕР[/bold green]", border_style="green"))
            console.print("")  # Пустая строка чтобы курсор был на чистом месте

        # Пауза чтобы терминал успел отрисовать панель перед следующим вводом
        import time
        time.sleep(0.1)

if __name__ == "__main__":
    main()