import sqlite3
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()
DB_FILE = "ttrpg_database.db"

def get_data(query):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except sqlite3.OperationalError:
        return None

def show_npcs():
    rows = get_data("SELECT name, lore_and_stats FROM npcs")
    table = Table(title="👥 База NPC", show_header=True, header_style="bold yellow", border_style="yellow")
    table.add_column("Имя", style="cyan", no_wrap=True)
    table.add_column("Данные (Lore, Stats, MBTI)", style="white")

    if rows:
        for name, info in rows:
            table.add_row(name, info)
        console.print(table)
    else:
        console.print("[yellow]Таблица NPC пуста или еще не создана.[/yellow]")

def show_locations():
    rows = get_data("SELECT name, description FROM locations")
    table = Table(title="📍 База Локаций", show_header=True, header_style="bold blue", border_style="blue")
    table.add_column("Название", style="cyan", no_wrap=True)
    table.add_column("Описание и Лор", style="white")

    if rows:
        for name, desc in rows:
            table.add_row(name, desc)
        console.print(table)
    else:
        console.print("[yellow]Таблица Локаций пуста.[/yellow]")

def show_summaries():
    rows = get_data("SELECT id, content FROM summaries")
    if rows:
        console.print("\n[bold gold1]📜 ИСТОРИЯ СЕССИЙ (САММАРИ):[/bold gold1]")
        for sid, content in rows:
            console.print(Panel(Markdown(content), title=f"Сессия №{sid}", border_style="gold1"))
    else:
        console.print("[yellow]История сессий пуста.[/yellow]")

def main():
    console.print(Panel.fit("🔍 ИНСПЕКТОР БАЗЫ ДАННЫХ TTRPG", style="bold magenta"))
    
    # Проверка существования файла
    import os
    if not os.path.exists(DB_FILE):
        console.print(f"[bold red]Ошибка:[/bold red] Файл базы данных '{DB_FILE}' не найден. Сначала запустите игру.")
        return

    show_npcs()
    console.print("\n")
    show_locations()
    console.print("\n")
    show_summaries()

if __name__ == "__main__":
    main()