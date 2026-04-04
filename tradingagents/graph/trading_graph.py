# tradingagents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# 从 agent_utils 导入抽象化工具方法
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_company_announcements,
    get_market_news,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """
        初始化交易代理图及相关组件。
        
        参数：
            selected_analysts: 已启用的分析师标识列表。
            debug: 是否以调试模式运行工作流。
            config: 运行时配置映射。
            callbacks: 执行期间使用的可选回调处理器。
        
        返回：
            None: 无返回值。
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # 更新数据接口配置
        set_config(self.config)

        # 创建必要目录
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # 按模型提供方的思考配置初始化 LLM
        llm_kwargs = self._get_provider_kwargs()

        # 如果存在回调，则附加到构造参数中传给 LLM
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        # 初始化记忆组件
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # 创建工具节点
        self.tool_nodes = self._create_tool_nodes()

        # 初始化各组件
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.portfolio_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # 状态跟踪
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # 日期到完整状态字典的映射

        # 构建图结构
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """
        获取创建 LLM 客户端时所需的提供方专属参数。
        
        返回：
            Dict[str, Any]: 处理后的参数字典。
        """
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "azure":
            api_version = self.config.get("azure_api_version")
            if api_version:
                kwargs["azure_api_version"] = api_version
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            content_filter_max_retries = self.config.get("content_filter_max_retries")
            if content_filter_max_retries is not None:
                kwargs["content_filter_max_retries"] = content_filter_max_retries
            content_filter_skip_message = self.config.get("content_filter_skip_message")
            if content_filter_skip_message:
                kwargs["content_filter_skip_message"] = content_filter_skip_message

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        # 所有提供方统一注入超时配置
        timeout = self.config.get("timeout")
        if timeout is not None:
            kwargs["timeout"] = timeout

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """
        基于抽象工具方法创建不同数据源的工具节点。
        
        返回：
            Dict[str, ToolNode]: 构建完成的工具节点映射。
        """
        return {
            "market": ToolNode(
                [
                    # 核心行情工具
                    get_stock_data,
                    # 技术指标工具
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # 面向社交分析的资讯工具
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    get_news,
                    get_market_news,
                    get_company_announcements,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # 基本面分析工具
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """
        在指定日期为某个标的执行交易代理图。
        
        参数：
            company_name: 传入图工作流的股票代码或公司标识。
            trade_date: YYYY-MM-DD 格式的交易日期。
        
        返回：
            None: 无返回值。
        """

        self.ticker = company_name

        # 初始化状态
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # 调试模式：保留完整追踪
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # 标准模式：直接执行，不保留追踪
            final_state = self.graph.invoke(init_agent_state, **args)

        # 保存当前状态以供反思模块使用
        self.curr_state = final_state

        # 记录状态
        self._log_state(trade_date, final_state)

        # 返回决策结果及处理后的信号
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """
        将最终状态记录到 JSON 文件。
        
        参数：
            trade_date: YYYY-MM-DD 格式的交易日期。
            final_state: 工作流执行完成后的最终状态。
        
        返回：
            None: 无返回值。
        """
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "final_market_report": final_state.get("final_market_report", ""),
            "final_sentiment_report": final_state.get("final_sentiment_report", ""),
            "final_news_report": final_state.get("final_news_report", ""),
            "final_fundamentals_report": final_state.get("final_fundamentals_report", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "latest_speaker": final_state["investment_debate_state"][
                    "latest_speaker"
                ],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_investment_plan_report": final_state.get("final_investment_plan_report", ""),
            "final_trader_investment_plan_report": final_state.get("final_trader_investment_plan_report", ""),
            "final_trade_decision": final_state["final_trade_decision"],
            "final_trade_decision_report": final_state.get("final_trade_decision_report", ""),
        }

        # 保存到文件
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """
        基于收益结果反思决策并更新记忆。
        
        参数：
            returns_losses: 用于反思的收益或盈亏结果。
        
        返回：
            None: 无返回值。
        """
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """
        处理信号并提取核心决策。
        
        参数：
            full_signal: 完整交易信号文本。
        
        返回：
            None: 无返回值。
        """
        return self.signal_processor.process_signal(full_signal)
