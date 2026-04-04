from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG, build_runtime_config, save_last_config
from cli.models import AnalystType
from cli.utils import *
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler

console = Console()

DEFAULT_ANALYSTS = [
    AnalystType.MARKET,
    AnalystType.SOCIAL,
    AnalystType.NEWS,
    AnalystType.FUNDAMENTALS,
]


def parse_selected_analysts(config) -> list[AnalystType]:
    """
    将配置中的分析师列表转换为 `AnalystType` 列表。

    参数：
        config: 运行时配置映射。

    返回：
        list[AnalystType]: 规范化后的分析师枚举列表。
    """
    configured = config.get("selected_analysts", [analyst.value for analyst in DEFAULT_ANALYSTS])
    parsed = []
    for analyst in configured:
        try:
            parsed.append(AnalystType(str(analyst).lower()))
        except ValueError:
            continue
    return parsed or list(DEFAULT_ANALYSTS)

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# 使用 deque 保存最近消息，并限制最大长度
class MessageBuffer:
    # 固定执行的团队，不允许用户取消
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager", "Report Finalizer"],
    }

    # 分析师名称映射
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Social Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # 报告分段映射：section -> (控制该分段的 analyst_key, 完成该分段的最终代理)
    # analyst_key：用于控制该分段是否启用的分析师键（None 表示始终包含）
    # finalizing_agent：只有对应代理完成后，该分段才算完成
    REPORT_SECTIONS = {
        "final_market_report": ("market", "Report Finalizer"),
        "final_sentiment_report": ("social", "Report Finalizer"),
        "final_news_report": ("news", "Report Finalizer"),
        "final_fundamentals_report": ("fundamentals", "Report Finalizer"),
        "final_investment_plan_report": (None, "Report Finalizer"),
        "final_trader_investment_plan_report": (None, "Report Finalizer"),
        "final_trade_decision_report": (None, "Report Finalizer"),
    }

    def __init__(self, max_length=100):
        """
        初始化对象。
        
        参数：
            max_length: 最大缓存长度。
        
        返回：
            None: 无返回值。
        """
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # 保存完整的最终报告
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._last_message_id = None

    def init_for_analysis(self, selected_analysts):
        """
        根据已选分析师初始化代理状态与报告分段。
        
        参数：
            selected_analysts: 已启用的分析师标识列表。
        
        返回：
            None: 无返回值。
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # 动态构建 agent_status
        self.agent_status = {}

        # 加入已选择的分析师
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # 加入固定团队成员
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # 动态构建 report_sections
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # 重置其他状态
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._last_message_id = None

    def get_completed_reports_count(self):
        """
        统计已完成的报告数量（对应最终代理需已完成）。
        
        返回：
            Any: 当前查询结果。
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # 仅当分段已有内容且对应最终代理完成时，才视为完成
            has_content = bool(self.report_sections.get(section))
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        """
        追加消息。
        
        参数：
            message_type: 消息类型。
            content: 需要展示或规范化的消息内容。
        
        返回：
            None: 无返回值。
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        """
        记录工具调用。
        
        参数：
            tool_name: 工具名称。
            args: 透传给底层可调用对象的位置参数。
        
        返回：
            None: 无返回值。
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        """
        更新代理状态。
        
        参数：
            agent: 代理名称。
            status: 要设置的状态文本。
        
        返回：
            None: 无返回值。
        """
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        """
        更新报告分段。
        
        参数：
            section_name: 报告分段名称。
            content: 需要展示或规范化的消息内容。
        
        返回：
            None: 无返回值。
        """
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # 面板中仅展示最近一次更新的报告分段
        """
        刷新当前报告。
        
        返回：
            None: 无返回值。
        """
        latest_section = None
        latest_content = None

        # 找到最近更新的分段
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # 格式化当前分段用于展示
            section_titles = {
                "final_market_report": "Market Analysis",
                "final_sentiment_report": "Social Sentiment",
                "final_news_report": "News Analysis",
                "final_fundamentals_report": "Fundamentals Analysis",
                "final_investment_plan_report": "Research Team Decision",
                "final_trader_investment_plan_report": "Trading Team Plan",
                "final_trade_decision_report": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # 更新最终完整报告
        self._update_final_report()

    def _update_final_report(self):
        """
        刷新最终报告。
        
        返回：
            None: 无返回值。
        """
        report_parts = []

        analyst_sections = [
            "final_market_report",
            "final_sentiment_report",
            "final_news_report",
            "final_fundamentals_report",
        ]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("final_market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['final_market_report']}"
                )
            if self.report_sections.get("final_sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['final_sentiment_report']}"
                )
            if self.report_sections.get("final_news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['final_news_report']}"
                )
            if self.report_sections.get("final_fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['final_fundamentals_report']}"
                )

        if self.report_sections.get("final_investment_plan_report"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['final_investment_plan_report']}")

        if self.report_sections.get("final_trader_investment_plan_report"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['final_trader_investment_plan_report']}")

        if self.report_sections.get("final_trade_decision_report"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision_report']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    """
    创建并返回界面布局。
    
    返回：
        Callable | object: 当前组件生成的可调用对象或实例。
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """
    格式化用于展示的 token 数量。
    
    参数：
        n: 需要格式化的数值。
    
    返回：
        str: 格式化后的 token 展示文本。
    """
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    """
    更新界面显示。
    
    参数：
        layout: Rich layout object to update.
        spinner_text: Optional spinner text shown in the CLI layout.
        stats_handler: Statistics handler used to display token and cost data.
        start_time: Start timestamp used to compute elapsed runtime.
    
    返回：
        None: 无返回值。
    """
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # 只保留在 agent_status 中存在的团队成员
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # 合并工具调用与消息
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # 取前 N 条消息（最新优先）
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # 对内容进行自动换行格式化
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # 组装统计信息片段
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """
    在启动分析展示前获取全部用户选择。
    
    返回：
        Any: 当前查询结果。
    """
    # 展示 ASCII 艺术欢迎信息
    with open(Path(__file__).parent / "static" / "welcome.txt", "r", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM A-Share Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )
    welcome_content += "\n"
    welcome_content += (
        "[dim]Modified by [Michael Yuan](https://michaelyuancb.github.io/)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM A-Share Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # 拉取并展示公告，失败时静默处理
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # 为每个步骤创建带边框的问题框
    def create_question_box(title, prompt, default=None):
        """
        创建并返回问题框。
        
        参数：
            title: 表格标题。
            prompt: 需要展示的提示文本。
            default: 可选的默认值。
        
        返回：
            Callable | object: 当前组件生成的可调用对象或实例。
        """
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the exact A-share ticker to analyze (examples: 600519, 000001, 300750, 688041)",
            "600519",
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    current_config = build_runtime_config()

    internal_language = current_config.get("internal_language", "English")
    output_language = current_config.get(
        "final_output_language",
        current_config.get("output_language", "Chinese"),
    )
    selected_analysts = parse_selected_analysts(current_config)
    selected_research_depth = current_config.get(
        "research_depth",
        current_config.get("max_debate_rounds", 1),
    )
    selected_llm_provider = current_config.get("llm_provider", "openai")
    backend_url = current_config.get("backend_url", "")
    selected_shallow_thinker = current_config.get("quick_think_llm", "")
    selected_deep_thinker = current_config.get("deep_think_llm", "")
    azure_api_version = current_config.get("azure_api_version")
    thinking_level = current_config.get("google_thinking_level")
    reasoning_effort = current_config.get("openai_reasoning_effort")
    anthropic_effort = current_config.get("anthropic_effort")

    def format_optional(value):
        """
        格式化可选配置值。

        参数：
            value: 需要展示的配置值。

        返回：
            str: 格式化后的配置文本。
        """
        if value in (None, "", []):
            return "未设置"
        return str(value)

    def build_config_summary() -> str:
        """
        构建当前默认配置摘要文本。

        返回：
            str: 汇总后的配置文本。
        """
        summary_lines = [
            f"[bold]Internal Language:[/bold] {internal_language}",
            f"[bold]Final Output Language:[/bold] {output_language}",
            f"[bold]Analysts:[/bold] {', '.join(analyst.value for analyst in selected_analysts)}",
            f"[bold]Research Depth:[/bold] {selected_research_depth}",
            f"[bold]LLM Provider:[/bold] {selected_llm_provider}",
            f"[bold]Backend URL:[/bold] {format_optional(backend_url)}",
            f"[bold]Quick-Thinking Model:[/bold] {format_optional(selected_shallow_thinker)}",
            f"[bold]Deep-Thinking Model:[/bold] {format_optional(selected_deep_thinker)}",
        ]

        provider_lower = selected_llm_provider.lower()
        if provider_lower == "azure":
            summary_lines.append(f"[bold]Azure API Version:[/bold] {format_optional(azure_api_version)}")
            summary_lines.append(f"[bold]Reasoning Effort:[/bold] {format_optional(reasoning_effort)}")
        elif provider_lower == "openai":
            summary_lines.append(f"[bold]Reasoning Effort:[/bold] {format_optional(reasoning_effort)}")
        elif provider_lower == "google":
            summary_lines.append(f"[bold]Thinking Mode:[/bold] {format_optional(thinking_level)}")
        elif provider_lower == "anthropic":
            summary_lines.append(f"[bold]Effort Level:[/bold] {format_optional(anthropic_effort)}")

        return "\n".join(summary_lines)

    console.print(
        Panel(
            build_config_summary(),
            border_style="yellow",
            padding=(1, 2),
            title="Current Configuration",
            subtitle="Confirm to continue, or choose to modify",
        )
    )

    modify_config = typer.confirm("Modify configuration?", default=False)
    if not modify_config:
        confirm_config = typer.confirm("Use the configuration above?", default=True)
        if not confirm_config:
            modify_config = True

    if modify_config:
        # Step 3: Internal language
        console.print(
            create_question_box(
                "Step 3: Internal Language",
                "Select the language used for internal analysis, debate, and collaboration"
            )
        )
        internal_language = ask_internal_language()

        # Step 4: Final output language
        console.print(
            create_question_box(
                "Step 4: Final Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language()

        # 第 5 步：选择分析师
        console.print(
            create_question_box(
                "Step 5: Analysts Team", "Select your LLM analyst agents for the analysis"
            )
        )
        selected_analysts = select_analysts()
        console.print(
            f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
        )

        # Step 6: Research depth
        console.print(
            create_question_box(
                "Step 6: Research Depth", "Select your research depth level"
            )
        )
        selected_research_depth = select_research_depth()

        # Step 7: LLM Provider
        console.print(
            create_question_box(
                "Step 7: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider()

        # Step 8: Thinking agents / Azure deployments
        provider_lower = selected_llm_provider.lower()
        azure_api_version = None
        thinking_level = None
        reasoning_effort = None
        anthropic_effort = None

        if provider_lower == "azure":
            console.print(
                create_question_box(
                    "Step 8: Azure Connection", "Configure Azure OpenAI endpoint and API version"
                )
            )
            default_endpoint = current_config.get("backend_url", "") if current_config.get("llm_provider") == "azure" else ""
            backend_url = ask_azure_endpoint(default=default_endpoint)
            azure_api_version = ask_azure_api_version(
                default=current_config.get("azure_api_version", "2024-12-01-preview")
            )

            console.print(
                create_question_box(
                    "Step 9: Azure Deployments", "Enter the Azure deployment names used for analysis"
                )
            )
            selected_shallow_thinker = ask_azure_deployment(
                "quick thinking",
                default=current_config.get("quick_think_llm", ""),
            )
            selected_deep_thinker = ask_azure_deployment(
                "deep thinking",
                default=current_config.get("deep_think_llm", ""),
            )
        else:
            console.print(
                create_question_box(
                    "Step 8: Thinking Agents", "Select your thinking agents for analysis"
                )
            )
            selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
            selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

        # Step 9/10: Provider-specific thinking configuration
        if provider_lower == "google":
            console.print(
                create_question_box(
                    "Step 9: Thinking Mode",
                    "Configure Gemini thinking mode"
                )
            )
            thinking_level = ask_gemini_thinking_config()
        elif provider_lower in ("openai", "azure"):
            console.print(
                create_question_box(
                    "Step 9: Reasoning Effort",
                    "Configure reasoning effort level"
                )
            )
            reasoning_effort = ask_openai_reasoning_effort()
        elif provider_lower == "anthropic":
            console.print(
                create_question_box(
                    "Step 9: Effort Level",
                    "Configure Claude effort level"
                )
            )
            anthropic_effort = ask_anthropic_effort()

    selections = {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "azure_api_version": azure_api_version,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "internal_language": internal_language,
        "output_language": output_language,
        "final_output_language": output_language,
    }

    if modify_config:
        persisted_config = current_config.copy()
        persisted_config.update(
            {
                "internal_language": internal_language,
                "final_output_language": output_language,
                "output_language": output_language,
                "selected_analysts": [analyst.value for analyst in selected_analysts],
                "research_depth": selected_research_depth,
                "max_debate_rounds": selected_research_depth,
                "max_risk_discuss_rounds": selected_research_depth,
                "llm_provider": selected_llm_provider.lower(),
                "backend_url": backend_url,
                "quick_think_llm": selected_shallow_thinker,
                "deep_think_llm": selected_deep_thinker,
                "azure_api_version": azure_api_version,
                "google_thinking_level": thinking_level,
                "openai_reasoning_effort": reasoning_effort,
                "anthropic_effort": anthropic_effort,
            }
        )
        save_last_config(persisted_config)

    return selections


def get_ticker():
    """
    从用户输入中获取股票代码。
    
    返回：
        Any: 当前查询结果。
    """
    return typer.prompt("", default="600519")


def get_analysis_date():
    """
    从用户输入中获取分析日期。
    
    返回：
        Any: 当前查询结果。
    """
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # 校验日期格式，并确保日期不晚于当前时间
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """
    将完整分析报告按目录结构保存到磁盘。
    
    参数：
        final_state: Final graph state produced by the workflow.
        ticker: 待分析公司的 A 股股票代码。
        save_path: 输出文件的可选保存路径。
    
    返回：
        None: 无返回值。
    """
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("final_market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market_report.md").write_text(final_state["final_market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["final_market_report"]))
    if final_state.get("final_sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment_report.md").write_text(final_state["final_sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Social Analyst", final_state["final_sentiment_report"]))
    if final_state.get("final_news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news_report.md").write_text(final_state["final_news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["final_news_report"]))
    if final_state.get("final_fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals_report.md").write_text(final_state["final_fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["final_fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    if final_state.get("final_investment_plan_report"):
        research_dir = save_path / "2_research"
        research_dir.mkdir(exist_ok=True)
        (research_dir / "investment_plan.md").write_text(final_state["final_investment_plan_report"], encoding="utf-8")
        sections.append(f"## II. Research Team Decision\n\n{final_state['final_investment_plan_report']}")

    if final_state.get("final_trader_investment_plan_report"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader_investment_plan_report.md").write_text(final_state["final_trader_investment_plan_report"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n{final_state['final_trader_investment_plan_report']}")

    if final_state.get("final_trade_decision_report"):
        portfolio_dir = save_path / "4_portfolio"
        portfolio_dir.mkdir(exist_ok=True)
        (portfolio_dir / "final_trade_decision_report.md").write_text(final_state["final_trade_decision_report"], encoding="utf-8")
        sections.append(f"## IV. Portfolio Management Decision\n\n{final_state['final_trade_decision_report']}")

    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"


def display_complete_report(final_state):
    """
    顺序展示完整分析报告，避免终端截断。
    
    参数：
        final_state: Final graph state produced by the workflow.
    
    返回：
        None: 无返回值。
    """
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    analysts = []
    if final_state.get("final_market_report"):
        analysts.append(("Market Analyst", final_state["final_market_report"]))
    if final_state.get("final_sentiment_report"):
        analysts.append(("Social Analyst", final_state["final_sentiment_report"]))
    if final_state.get("final_news_report"):
        analysts.append(("News Analyst", final_state["final_news_report"]))
    if final_state.get("final_fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["final_fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    if final_state.get("final_investment_plan_report"):
        console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
        console.print(Panel(Markdown(final_state["final_investment_plan_report"]), title="Research Manager", border_style="blue", padding=(1, 2)))

    if final_state.get("final_trader_investment_plan_report"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["final_trader_investment_plan_report"]), title="Trader", border_style="blue", padding=(1, 2)))

    if final_state.get("final_trade_decision_report"):
        console.print(Panel("[bold]IV. Portfolio Manager Decision[/bold]", border_style="green"))
        console.print(Panel(Markdown(final_state["final_trade_decision_report"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """
    更新研究团队成员状态（不含 Trader）。
    
    参数：
        status: 要设置的状态文本。
    
    返回：
        None: 无返回值。
    """
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk):
    """
    根据累计报告状态更新分析师状态。
    
    参数：
        message_buffer: Shared message buffer used by the CLI display.
        chunk: Incremental graph output chunk to inspect.
    
    返回：
        None: 无返回值。
    """
    selected = message_buffer.selected_analysts
    found_active = False

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        has_report = bool(
            chunk.get(report_key)
            or message_buffer.agent_status.get(agent_name) == "completed"
        )

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """
    从不同消息格式中提取字符串内容。
    
    参数：
        content: 需要展示或规范化的消息内容。
    
    返回：
        Any: Extracted value derived from the provided input.
    """
    import ast

    def is_empty(val):
        """
        使用 Python 真值规则判断内容是否为空。
        
        参数：
            val: 待判断的值。
        
        返回：
            bool: 条件满足时返回 True，否则返回 False。
        """
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """
    识别 LangChain 消息类型并提取展示内容。
    
    参数：
        message: Message object to classify or inspect.
    
    返回：
        tuple[str, str | None]: Extracted value derived from the provided input.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """
    格式化终端展示所需的工具参数。
    
    参数：
        args: 透传给底层可调用对象的位置参数。
        max_length: 最大缓存长度。
    
    返回：
        str: 格式化后的字符串结果。
    """
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def run_analysis():
    # First get all user selections
    """
    执行分析流程。
    
    返回：
        None: 无返回值。
    """
    selections = get_user_selections()

    # Create config with selected research depth
    config = build_runtime_config()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["research_depth"] = selections["research_depth"]
    config["selected_analysts"] = [analyst.value for analyst in selections["analysts"]]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    config["azure_api_version"] = selections.get("azure_api_version")
    # 不同提供方的思考参数配置
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["internal_language"] = selections.get("internal_language", "English")
    config["output_language"] = selections.get(
        "final_output_language",
        selections.get("output_language", "Chinese"),
    )
    config["final_output_language"] = selections.get(
        "final_output_language",
        selections.get("output_language", "Chinese"),
    )

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # 将分析师选择归一到预定义顺序（输入是 set，输出顺序固定）
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # 使用绑定了回调的 LLM 初始化图结构
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # 用已选分析师初始化消息缓冲区
    message_buffer.init_for_analysis(selected_analyst_keys)

    # 记录开始时间，用于展示耗时
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        """
        保存消息装饰器。
        
        参数：
            obj: 待包装的对象实例。
            func_name: 需要包装的方法名。
        
        返回：
            None: 无返回值。
        """
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            执行包装函数。
            
            参数：
                args: 透传给底层可调用对象的位置参数。
                kwargs: 透传给底层可调用对象的关键字参数。
            
            返回：
                None: 无返回值。
            """
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # 将换行替换为空格
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        """
        保存工具调用装饰器。
        
        参数：
            obj: 待包装的对象实例。
            func_name: 需要包装的方法名。
        
        返回：
            None: 无返回值。
        """
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            执行包装函数。
            
            参数：
                args: 透传给底层可调用对象的位置参数。
                kwargs: 透传给底层可调用对象的关键字参数。
            
            返回：
                None: 无返回值。
            """
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        """
        保存报告分段装饰器。
        
        参数：
            obj: 待包装的对象实例。
            func_name: 需要包装的方法名。
        
        返回：
            None: 无返回值。
        """
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            """
            执行包装函数。
            
            参数：
                section_name: 报告分段名称。
                content: 需要展示或规范化的消息内容。
            
            返回：
                None: 无返回值。
            """
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / f"{section_name}.md", "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # 将首位分析师状态更新为 in_progress
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # 初始化状态，并获取携带回调的图调用参数
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # 如存在消息则处理，并基于消息 ID 跳过重复项
            if len(chunk["messages"]) > 0:
                last_message = chunk["messages"][-1]
                msg_id = getattr(last_message, "id", None)

                if msg_id != message_buffer._last_message_id:
                    message_buffer._last_message_id = msg_id

                    # Add message to buffer
                    msg_type, content = classify_message_type(last_message)
                    if content and content.strip():
                        message_buffer.add_message(msg_type, content)

                    # 处理工具调用
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if isinstance(tool_call, dict):
                                message_buffer.add_tool_call(
                                    tool_call["name"], tool_call["args"]
                                )
                            else:
                                message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # 根据报告状态更新分析师状态（每个数据块都会执行）
            update_analyst_statuses(message_buffer, chunk)

            # 研究团队：处理投资辩论状态
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if judge:
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # 风险管理团队：处理风险辩论状态
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                if judge:
                    if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")
                        message_buffer.update_agent_status("Report Finalizer", "in_progress")

            for section in message_buffer.report_sections.keys():
                if chunk.get(section):
                    message_buffer.update_report_section(section, chunk[section])

            if any(chunk.get(section) for section in message_buffer.report_sections):
                message_buffer.update_agent_status("Report Finalizer", "completed")

            # 更新界面显示
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # 获取最终状态与决策
        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        # 将所有代理状态更新为 completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )

        # 更新最终报告分段
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


@app.command()
def analyze():
    """
    执行分析命令。
    
    返回：
        None: 无返回值。
    """
    run_analysis()


if __name__ == "__main__":
    app()
