"""OfficeBench sub-agents for individual applications."""

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

import apps
from benchmarks.officebench.langchain_adapter import LangChainSubagentAdapter
from benchmarks.officebench.prompts import AGENT_SYSTEM_PROMPT, APP_EXTRA_INSTRUCTIONS
from benchmarks.officebench.tools import make_interact_tool
from benchmarks.shared.cards import build_agent_configs, load_cards
from runtime.llm import create_llm

DEFAULT_CARDS_PATH = "agents/officebench/rich/cards.json"


def _normalise_excluded_actions(
    excluded_actions=None,
) -> dict[str, set[str]]:
    if not excluded_actions:
        return {}

    items = (
        excluded_actions.items()
        if hasattr(excluded_actions, "items")
        else vars(excluded_actions).items()
    )
    normalised: dict[str, set[str]] = {}
    for app, actions in items:
        if isinstance(actions, str):
            actions = [actions]
        if actions:
            normalised[str(app)] = {str(action) for action in actions}
    return normalised


def _filter_actions_for_agent(agent_name: str, excluded_actions: dict[str, set[str]]) -> dict:
    """Filter available actions for an agent based on excluded actions config."""
    excluded = excluded_actions.get(agent_name, set())
    return {
        action_name: action
        for action_name, action in apps.AVAILABLE_ACTIONS.get(agent_name, {}).items()
        if action_name not in excluded
    }


class OfficeBenchAgents:
    """Build specialized OfficeBench sub-agents and their registry configs."""

    def __init__(
        self,
        env,
        task_config=None,
        model_name: str | None = None,
        temperature: float = 0.0,
        request_timeout: float | None = 120.0,
        excluded_actions: dict[str, set[str] | list[str] | str] | None = None,
        cards_path: str | None = None,
        extra_cards_path: str | None = None,
    ):
        self.excluded_actions = _normalise_excluded_actions(excluded_actions)
        agent_tool = make_interact_tool(env, excluded_actions=self.excluded_actions)
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout

        agent_cards = load_cards(cards_path or DEFAULT_CARDS_PATH, extra_cards_path)
        active_agent_names = [
            agent_name
            for agent_name in agent_cards
            if agent_name in env.available_agents and agent_name in env.available_apps
        ]
        self.subagent_graphs = {
            agent_name: self._create_subagent(
                agent_name,
                task_config,
                agent_tool,
            )
            for agent_name in active_agent_names
        }
        self.agent_configs = build_agent_configs(agent_cards, self.subagent_graphs)

    def _create_subagent(self, agent_name: str, task_config: dict, agent_tool):
        """Create a specialized subagent for a specific application.
        
        Args:
            agent_name: Name of the agent (e.g., 'calendar', 'email')
            task_config: Task configuration for prompt context
            agent_tool: Tool for agent interaction with the environment
            
        Returns:
            LangChain agent graph for the subagent
        """
        system_prompt = self.construct_subagent_system_message(task_config, agent_name)
        # Disable parallel tool calls to mitigate rate limit issues
        llm = create_llm(
            model_name=self.model_name,
            temperature=self.temperature,
            model_kwargs={"parallel_tool_calls": False},
            request_timeout=self.request_timeout,
        )

        subagent_graph = create_agent(
            model=llm,
            tools=[agent_tool],
            system_prompt=SystemMessage(content=system_prompt),
            name=f"{agent_name}_subagent",
        )

        return LangChainSubagentAdapter(subagent_graph, agent_name=agent_name)

    def _extract_prompt_context(self, task_config: dict) -> dict:
        """Extract context variables from config for prompts.
        
        Args:
            task_config: Task configuration dictionary
            
        Returns:
            Dictionary with username, date, weekday, time
        """
        return {
            'username': task_config['username'],
            'date': task_config['date'],
            'weekday': task_config['weekday'],
            'time': task_config['time']
        }

    def construct_subagent_system_message(self, task_config: dict, agent_name: str) -> str:
        """Build system message for a specialized subagent.
        
        Args:
            task_config: Task configuration for context
            agent_name: Name of the agent
            
        Returns:
            Formatted system prompt string
        """
        context = self._extract_prompt_context(task_config)

        actions = _filter_actions_for_agent(agent_name, self.excluded_actions)
        detailed = "\n".join(
            f"- {action.DEMO}"
            for action in actions.values()
        )

        prompt = AGENT_SYSTEM_PROMPT.format(
            **context,
            app_name=agent_name,
            detailed=detailed
        ).strip()

        if agent_name in APP_EXTRA_INSTRUCTIONS:
            prompt += "\n" + APP_EXTRA_INSTRUCTIONS[agent_name].strip()

        return prompt

    def get_all_agent_configs(self) -> dict:
        """Get all agent configurations.
        
        Returns:
            Dictionary mapping agent names to their configurations
        """
        return self.agent_configs
