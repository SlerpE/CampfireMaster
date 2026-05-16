import os
import sqlite3
import operator
from typing import Annotated, TypedDict, Sequence, List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.theme import Theme

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. НАСТРОЙКА ИНТЕРФЕЙСА (RICH)
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
# 2. ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ (SQLITE)
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
# 3. НАСТРОЙКА LLM
# ==========================================
llm = ChatOpenAI(
    base_url="http://192.168.2.107:5001/v1",
    api_key="local-key",
    model="Gemma-4-MeroMero",
    temperature=0.7,
    max_tokens=2000
)

# ==========================================
# 4. СКИЛЛ (ТОЛЬКО PYTHON)
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
    screen_context: str
    locator_context: str
    npc_context: str
    is_session_end: bool
    active_agents: List[str]  # Список агентов, которых разбудит Оркестратор

# ==========================================
# 6. УЗЛЫ (АГЕНТЫ)
# ==========================================

def agent_orchestrator(state: AgentState):
    """Оркестратор решает, какие агенты нужны для обработки запроса."""
    if state.get("is_session_end"):
        return {"active_agents": ["summarizer"]}
        
    sys_prompt = SystemMessage(content="""Ты ИИ-Оркестратор. Твоя задача определить, какие модули нужны для ответа.
    Модули:
    - screen : нужны скрытые броски кубиков за кадром, планирование угрозы.
    - locator : игрок перемещается в новую локацию, осматривает местность.
    - npc : игрок говорит с персонажем, просит создать персонажа, упоминает живых существ.
    Выведи ТОЛЬКО названия нужных модулей. Пример: npc, screen""")
    
    response = llm.invoke([sys_prompt, state["messages"][-1]])
    content = response.content.lower()
    
    active = []
    if "screen" in content: active.append("screen")
    if "locator" in content: active.append("locator")
    if "npc" in content: active.append("npc")
    
    # Сбрасываем старые контексты и передаем список тех, кого надо запустить параллельно
    return {
        "active_agents": active,
        "screen_context": "",
        "locator_context": "",
        "npc_context": ""
    }

def agent_screen(state: AgentState):
    sys_prompt = SystemMessage(content="Ты Агент-Ширма. Напиши коротко свои скрытые мысли, планы на угрозу или проверки навыков.")
    response = llm.invoke([sys_prompt] + state["messages"][-2:])
    return {"screen_context": response.content}

def agent_locator(state: AgentState):
    locs = get_all("locations")
    db_text = "\n".join([f"[{row[0]}]: {row[1]}" for row in locs])
    sys_prompt = SystemMessage(content=f"""Ты Агент-Локатор. База локаций:\n{db_text}\nЕСЛИ СОЗДАЕШЬ/МЕНЯЕШЬ ЛОКАЦИЮ, начни ответ с: SAVE_LOC: Имя | Описание""")
    response = llm.invoke([sys_prompt] + state["messages"][-2:])
    content = response.content
    if "SAVE_LOC:" in content:
        try:
            parts = content.split("SAVE_LOC:")[1].split("|", 1)
            upsert_record("locations", parts[0].strip(), parts[1].strip())
            content = f"Локация {parts[0].strip()} сохранена. {parts[1].strip()}"
        except: pass
    return {"locator_context": content}

def agent_npc(state: AgentState):
    npcs = get_all("npcs")
    db_text = "\n".join([f"[{row[0]}]: {row[1]}" for row in npcs])
    sys_prompt = SystemMessage(content=f"""Ты Агент-Менеджер NPC. База NPC:\n{db_text}\nЕСЛИ СОЗДАЕШЬ/МЕНЯЕШЬ NPC, начни ответ с: SAVE_NPC: Имя | Описание, mbti, лор""")
    response = llm.invoke([sys_prompt] + state["messages"][-2:])
    content = response.content
    if "SAVE_NPC:" in content:
        try:
            parts = content.split("SAVE_NPC:")[1].split("|", 1)
            upsert_record("npcs", parts[0].strip(), parts[1].strip())
            content = f"NPC {parts[0].strip()} сохранен. {parts[1].strip()}"
        except: pass
    return {"npc_context": content}

def agent_master(state: AgentState):
    context = f"""
    [Ширма]: {state.get('screen_context', '')}
    [Локация]: {state.get('locator_context', '')}
    [NPC]: {state.get('npc_context', '')}
    """
    sys_prompt = SystemMessage(content=f"Ты Агент-Мастер (DM). Веди игру, используя контекст от параллельных модулей:\n{context}\nПри необходимости бросай кубики через python_interpreter.")
    response = llm_with_tools.invoke([sys_prompt] + state["messages"])
    return {"messages": [response]}

def agent_summarizer(state: AgentState):
    sys_prompt = SystemMessage(content="Ты Агент-Итогер. Сделай саммари сессии и распредели XP.")
    response = llm.invoke([sys_prompt] + state["messages"])
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
workflow.add_node("screen", agent_screen)
workflow.add_node("locator", agent_locator)
workflow.add_node("npc", agent_npc)
workflow.add_node("master", agent_master)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("summarizer", agent_summarizer)

workflow.set_entry_point("orchestrator")

# Из оркестратора возможны параллельные переходы
workflow.add_conditional_edges("orchestrator", route_from_orchestrator, ["screen", "locator", "npc", "master", "summarizer"])

# Все вспомогательные агенты после параллельной работы сходятся в Мастера (Fan-in)
workflow.add_edge("screen", "master")
workflow.add_edge("locator", "master")
workflow.add_edge("npc", "master")

# Условный переход от Мастера (тулзы или конец)
workflow.add_conditional_edges("master", should_continue_from_master, ["tools", END])
workflow.add_edge("tools", "master")
workflow.add_edge("summarizer", END)

app = workflow.compile()

# ==========================================
# 8. UI И ГЛАВНЫЙ ЦИКЛ
# ==========================================
def main():
    console.clear()
    console.print(Panel("[bold green]🐉 TTRPG AI Game Master (Parallel Routing Edition)[/bold green]\n* `/end` - итоги\n* `/quit` - выход", style="master"))
    
    past_history = ""
    summaries = get_all("summaries")
    if summaries:
        past_history = "История прошлых сессий:\n" + "\n".join([f"- Сессия {r[0]}: {r[1]}" for r in summaries])
        console.print("[info]База подтянута. Мастер помнит прошлые игры.[/info]")

    state: AgentState = {
        "messages": [SystemMessage(content=past_history)] if past_history else [],
        "screen_context": "", "locator_context": "", "npc_context": "",
        "is_session_end": False, "active_agents": []
    }

    while True:
        user_input = Prompt.ask("\n[bold cyan]Вы (Игрок)[/bold cyan]")
        if user_input.strip() == "/quit": break
        if user_input.strip() == "/end":
            state["is_session_end"] = True
            user_input = "СЕССИЯ ОКОНЧЕНА. Подведи итоги."
            
        state["messages"].append(HumanMessage(content=user_input))

        with console.status("[bold magenta]Оркестратор анализирует запрос...[/bold magenta]", spinner="dots") as status:
            for output in app.stream(state):
                for node_name, node_state in output.items():
                    if node_name == "orchestrator":
                        agents = node_state.get("active_agents", [])
                        msg = f"Оркестратор разбудил: {', '.join(agents)}" if agents else "Никто не нужен, сразу к Мастеру"
                        console.print(f"[dim info]⚙️ {msg}[/dim info]")
                        status.update("[bold magenta]Агенты шуршат параллельно...[/bold magenta]")
                        
                    elif node_name in ["screen", "locator", "npc"]:
                        console.print(f"[dim yellow]✓ {node_name.capitalize()} отработал(а)[/dim yellow]")
                        
                    elif node_name == "tools":
                        status.update("[bold red]Мастер кодит на Python...[/bold red]")
                        console.print(f"[bold red]⚙️ Python Output:[/bold red]\n{node_state['messages'][-1].content}")
                        
                    elif node_name == "master":
                        status.update("[bold green]Мастер пишет ответ...[/bold green]")

            last_msg = output[list(output.keys())[-1]]["messages"][-1]
            state["messages"].append(last_msg)

        if state["is_session_end"]:
            console.print(Panel(Markdown(last_msg.content), title="[bold gold1]ЛЕТОПИСЬ СЕССИИ[/bold gold1]", border_style="gold1"))
            break
        else:
            console.print(Panel(Markdown(last_msg.content), title="[bold green]Агент-Мастер[/bold green]", border_style="green"))

if __name__ == "__main__":
    main()