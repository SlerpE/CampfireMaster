import os
import sqlite3
from typing import Annotated, TypedDict, Sequence
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
import operator

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
    """Создает таблицы, если их нет"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Таблица NPC
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS npcs (
            name TEXT PRIMARY KEY,
            lore_and_stats TEXT
        )
    ''')
    
    # Таблица Локаций
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS locations (
            name TEXT PRIMARY KEY,
            description TEXT
        )
    ''')
    
    # Таблица Саммари сессий
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Вспомогательные функции для работы с БД
def get_all(table_name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    conn.close()
    return rows

def upsert_record(table_name, name, content):
    """Обновляет или вставляет новую запись (NPC/Локацию)"""
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
    """
    Выполняет Python код. Используй для математики и бросков кубиков.
    Возвращай результат через print().
    Пример:
    import random
    print(random.randint(1, 20))
    """
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

# ==========================================
# 6. УЗЛЫ (АГЕНТЫ)
# ==========================================

def agent_screen(state: AgentState):
    sys_prompt = SystemMessage(content="Ты Агент-Ширма. Скрыто планируешь действия за кадром, оцениваешь проверки навыков и угрозы. Напиши свои мысли коротко.")
    response = llm.invoke([sys_prompt] + state["messages"][-2:])
    return {"screen_context": response.content}

def agent_locator(state: AgentState):
    # Достаем все локации из SQLite
    locs = get_all("locations")
    db_text = "\n".join([f"[{row[0]}]: {row[1]}" for row in locs])
    
    sys_prompt = SystemMessage(content=f"""Ты Агент-Локатор. 
База локаций:\n{db_text}

Анализируй действия игрока. Если он в новой локации - придумай её (лор, вид). 
ЕСЛИ ТЫ СОЗДАЕШЬ ИЛИ МЕНЯЕШЬ ЛОКАЦИЮ, строго начни свой ответ с:
SAVE_LOC: Имя Локации | Описание
Иначе просто опиши текущую.""")
    
    response = llm.invoke([sys_prompt] + state["messages"][-2:])
    content = response.content
    
    # Парсим маркер для записи в SQLite
    if "SAVE_LOC:" in content:
        try:
            parts = content.split("SAVE_LOC:")[1].split("|", 1)
            loc_name = parts[0].strip()
            loc_desc = parts[1].strip()
            upsert_record("locations", loc_name, loc_desc)
            content = f"Локация {loc_name} загружена/обновлена в БД. {loc_desc}"
        except Exception:
            pass
            
    return {"locator_context": content}

def agent_npc(state: AgentState):
    # Достаем всех NPC из SQLite
    npcs = get_all("npcs")
    db_text = "\n".join([f"[{row[0]}]: {row[1]}" for row in npcs])
    
    sys_prompt = SystemMessage(content=f"""Ты Агент-Менеджер NPC.
База NPC:\n{db_text}

Если игрок обращается к новому NPC, инициализируй его (статблок, инвентарь, mbti, лор).
ЕСЛИ ТЫ СОЗДАЕШЬ ИЛИ ИЗМЕНЯЕШЬ NPC, строго начни свой ответ с:
SAVE_NPC: Имя Персонажа | Полное описание и статы
Иначе просто выдай инфу.""")
    
    response = llm.invoke([sys_prompt] + state["messages"][-2:])
    content = response.content
    
    # Парсим маркер для записи в SQLite
    if "SAVE_NPC:" in content:
        try:
            parts = content.split("SAVE_NPC:")[1].split("|", 1)
            npc_name = parts[0].strip()
            npc_lore = parts[1].strip()
            upsert_record("npcs", npc_name, npc_lore)
            content = f"NPC {npc_name} загружен/обновлен в БД. {npc_lore}"
        except Exception:
            pass

    return {"npc_context": content}

def agent_master(state: AgentState):
    context = f"""
    [Скрытые мысли Ширмы]: {state.get('screen_context', '')}
    [Данные о Локации]: {state.get('locator_context', '')}
    [Данные о NPC]: {state.get('npc_context', '')}
    """
    sys_prompt = SystemMessage(content=f"Ты Агент-Мастер (DM). Веди игру на основе контекста от модулей:\n{context}\nПри необходимости бросай кубики через python_interpreter.")
    
    response = llm_with_tools.invoke([sys_prompt] + state["messages"])
    return {"messages": [response]}

def agent_summarizer(state: AgentState):
    sys_prompt = SystemMessage(content="Ты Агент-Итогер. Сделай саммари сессии и распредели XP.")
    response = llm.invoke([sys_prompt] + state["messages"])
    
    # Сохраняем в SQLite
    save_summary(response.content)
    
    return {"messages": [AIMessage(content=f"\n*** ИТОГИ СЕССИИ ***\n{response.content}")]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if state.get("is_session_end"):
        return "summarizer"
    elif last_message.tool_calls:
        return "tools"
    return END

# ==========================================
# 7. СБОРКА ГРАФА
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("screen", agent_screen)
workflow.add_node("locator", agent_locator)
workflow.add_node("npc", agent_npc)
workflow.add_node("master", agent_master)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("summarizer", agent_summarizer)

workflow.set_entry_point("screen")
workflow.add_edge("screen", "locator")
workflow.add_edge("locator", "npc")
workflow.add_edge("npc", "master")
workflow.add_conditional_edges("master", should_continue, ["tools", "summarizer", END])
workflow.add_edge("tools", "master")
workflow.add_edge("summarizer", END)

app = workflow.compile()

# ==========================================
# 8. UI И ГЛАВНЫЙ ЦИКЛ
# ==========================================
def print_banner():
    console.clear()
    banner = Markdown("""
# 🐉 TTRPG AI Game Master
### LangGraph + Gemma-4 + SQLite + Rich
* Введите действие для игры.
* `/end` - завершить сессию и записать саммари в БД.
* `/quit` - выход.
    """)
    console.print(Panel(banner, style="master", border_style="bold yellow"))

def load_previous_summaries():
    summaries = get_all("summaries")
    if not summaries:
        return ""
    text = "История предыдущих сессий:\n"
    for row in summaries:
        text += f"--- Сессия {row[0]} ---\n{row[1]}\n"
    return text

def main():
    print_banner()
    
    past_history = load_previous_summaries()
    initial_context = ""
    if past_history:
        console.print(Panel("Загружены итоги из SQLite. Мастер помнит прошлое.", style="info"))
        initial_context = f"Учти события прошлых сессий:\n{past_history}"

    state: AgentState = {
        "messages": [SystemMessage(content=initial_context)] if initial_context else [],
        "screen_context": "",
        "locator_context": "",
        "npc_context": "",
        "is_session_end": False
    }

    while True:
        user_input = Prompt.ask("\n[bold cyan]Вы (Игрок)[/bold cyan]")
        
        if user_input.strip() == "/quit":
            console.print("[danger]Выход...[/danger]")
            break
            
        if user_input.strip() == "/end":
            state["is_session_end"] = True
            user_input = "СЕССИЯ ОКОНЧЕНА. Подведи итоги."
            
        state["messages"].append(HumanMessage(content=user_input))

        with console.status("[bold magenta]Агенты шуршат в SQLite...[/bold magenta]", spinner="dots") as status:
            for output in app.stream(state):
                for node_name, node_state in output.items():
                    if node_name == "screen":
                        status.update("[bold magenta]Ширма бросает кубы за кадром...[/bold magenta]")
                    elif node_name == "locator":
                        status.update("[bold blue]Локатор проверяет БД локаций...[/bold blue]")
                    elif node_name == "npc":
                        status.update("[bold yellow]Менеджер NPC обращается к БД...[/bold yellow]")
                    elif node_name == "tools":
                        status.update("[bold red]Мастер кодит на Python...[/bold red]")
                        last_msg = node_state["messages"][-1]
                        console.print(f"[bold red]⚙️ Python Output:[/bold red]\n{last_msg.content}")
                    elif node_name == "summarizer":
                        status.update("[bold green]Итогер пишет летопись в БД...[/bold green]")
                    elif node_name == "master":
                        status.update("[bold green]Мастер формулирует ответ...[/bold green]")

            last_msg = output[list(output.keys())[-1]]["messages"][-1]
            state["messages"].append(last_msg)

        if state["is_session_end"]:
            console.print(Panel(Markdown(last_msg.content), title="[bold gold1]ЛЕТОПИСЬ СЕССИИ[/bold gold1]", border_style="gold1"))
            break
        else:
            console.print(Panel(Markdown(last_msg.content), title="[bold green]Агент-Мастер[/bold green]", border_style="green"))

if __name__ == "__main__":
    main()