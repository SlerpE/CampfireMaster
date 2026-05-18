import sys
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# --- НАСТРОЙКИ ---
BASE_URL = "http://192.168.2.107:5001/v1"
MODEL_NAME = "Gemma-4-MeroMero"
API_KEY = "dummy-key" # Локальным серверам обычно плевать на ключ, но библиотека его просит

# Инициализация
console = Console()
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Системный промпт (можешь поменять под себя)
SYSTEM_PROMPT = "Ты полезный, умный и лаконичный ИИ-ассистент."

def init_history():
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def main():
    console.print(Panel.fit(
        f"🌟 Чат запущен!\nМодель: [bold green]{MODEL_NAME}[/]\nСервер: [bold blue]{BASE_URL}[/]\n\n"
        "Команды:\n"
        "[bold yellow]/exit[/] - выход\n"
        "[bold yellow]/clear[/] - очистить память (начать заново)", 
        title="Gemma Console Chat", border_style="cyan"
    ))

    history = init_history()

    while True:
        try:
            # Ввод пользователя
            console.print("\n[bold green]Вы:[/]")
            user_text = input("> ").strip()

            if not user_text:
                continue

            if user_text.lower() in ['/exit', '/quit', 'выход']:
                console.print("[bold red]До связи![/]")
                break
                
            if user_text.lower() in ['/clear', 'очистить']:
                history = init_history()
                console.print("[bold yellow]Память очищена. Контекст сброшен.[/]")
                continue

            # Добавляем сообщение в историю
            history.append({"role": "user", "content": user_text})

            # Отправка запроса с потоковым выводом (стриминг)
            console.print("\n[bold purple]Gemma:[/]")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,
                stream=True,
                temperature=0.7 # Можешь покрутить креативность (0.1 - 1.0)
            )

            full_reply = ""
            # Читаем ответ по кусочкам, чтобы не ждать окончания генерации
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text_chunk = chunk.choices[0].delta.content
                    full_reply += text_chunk
                    # Печатаем сразу же
                    sys.stdout.write(text_chunk)
                    sys.stdout.flush()
            
            print() # Перенос строки после завершения ответа

            # Сохраняем ответ модели в историю для поддержания диалога
            history.append({"role": "assistant", "content": full_reply})

        except KeyboardInterrupt:
            # Если нажать Ctrl+C
            console.print("\n[bold red]Чат прерван (Ctrl+C).[/]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Ошибка связи с сервером:[/] {e}")
            # Удаляем последнее сообщение юзера, чтобы оно не сломало контекст при ошибке
            if history[-1]["role"] == "user":
                history.pop()

if __name__ == "__main__":
    main()