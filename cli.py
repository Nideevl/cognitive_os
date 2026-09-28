import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.developer_agent import DeveloperAgent
from agent.key_manager import GroqKeyRotator
from dotenv import load_dotenv      

load_dotenv(Path(__file__).parent / ".env", override=False)   

console = Console()


# =====================================================================
# KEY LOADING
# =====================================================================
def load_groq_keys() -> list[str]:
    """Load Groq API keys from env.

    Priority:
      1. GROQ_API_KEYS   -> comma-separated single variable
      2. GROQ_API_KEY_1, GROQ_API_KEY_2, ... GROQ_API_KEY_N
      3. GROQ_API_KEY    -> single key (legacy fallback)
    """
    # 1. Comma-separated pool
    pool_env = os.getenv("GROQ_API_KEYS", "").strip()
    if pool_env:
        keys = [k.strip() for k in pool_env.split(",") if k.strip()]
        if keys:
            return keys

    # 2. Numbered keys (GROQ_API_KEY_1 ... GROQ_API_KEY_10)
    numbered = []
    for i in range(1, 11):
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if k:
            numbered.append(k)
    if numbered:
        return numbered

    # 3. Legacy single key
    single = os.getenv("GROQ_API_KEY", "").strip()
    if single:
        return [single]

    return []


# =====================================================================
# MODEL / PROJECT CONFIG
# =====================================================================
MODEL = "qwen/qwen3.8-27b"
PROJECT_ROOT = r"C:\Users\nidee\Downloads\work\resume\Car-Rent-Microservices"


def main():
    # ---- Load keys & build rotator ----
    groq_keys = load_groq_keys()
    if not groq_keys:
        console.print(Panel(
            "[bold red]No Groq API keys found.[/bold red]\n\n"
            "Set one of the following environment variables:\n"
            "  • [cyan]GROQ_API_KEYS[/cyan]   = 'gsk_aaa,gsk_bbb,gsk_ccc'  (comma-separated)\n"
            "  • [cyan]GROQ_API_KEY_1[/cyan], [cyan]GROQ_API_KEY_2[/cyan], ...  (numbered)\n"
            "  • [cyan]GROQ_API_KEY[/cyan]    = 'gsk_xxx'                   (single key)",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red"
        ))
        sys.exit(1)

    try:
        rotator = GroqKeyRotator(api_keys=groq_keys)
    except ValueError as e:
        console.print(f"[bold red]Key rotator init failed:[/bold red] {e}")
        sys.exit(1)

    console.print(
        f"[dim]Loaded [bold]{rotator.total_keys}[/bold] Groq API key(s) "
        f"into the rotation pool.[/dim]"
    )

    # ---- Boot agent ----
    agent = DeveloperAgent(PROJECT_ROOT, key_rotator=rotator, model=MODEL)

    console.print(Panel(
        f"[bold cyan]Cognitive OS: Progressive Developer Agent[/bold cyan]\n"
        f"Target Repo: [yellow]{Path(PROJECT_ROOT).name}[/yellow]\n"
        f"Available Services: {', '.join(agent.list_services())}\n"
        f"Key Pool Size: [bold]{rotator.total_keys}[/bold]",
        border_style="cyan"
    ))

    # ---- REPL ----
    while True:
        try:
            user_input = console.input(
                "\n[bold cyan]CognitiveOS[/bold cyan] [bold white]>[/bold white] "
            ).strip()

            if not user_input or user_input.lower() in ("q", "quit", "exit"):
                break

            response = agent.investigate(user_input)

            console.print("\n")
            console.print(Panel(
                Markdown(response),
                title="[bold green]Architectural Synthesis[/bold green]",
                border_style="green"
            ))
            console.print("─" * 60)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Exiting.[/dim]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Agent error:[/bold red] {e}")
            console.print("[dim]Continuing…[/dim]")


if __name__ == "__main__":
    main()