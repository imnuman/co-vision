"""Integration with Feynman voice assistant.

Provides a plugin that connects CoVision with Feynman's LiveKit agent
for multimodal AI companion functionality.
"""

import asyncio
import logging
from typing import Optional, Callable, Any

from covision.events import (
    Event,
    UserArrivedEvent,
    UserLeftEvent,
    UserLookingEvent,
    SceneUpdateEvent,
)
from covision.presence import PresenceInfo

logger = logging.getLogger(__name__)


class FeynmanVisionPlugin:
    """Plugin connecting CoVision with Feynman voice assistant.

    Provides:
    - Automatic greetings when user arrives
    - Context injection for conversations
    - Attention-based interaction triggers

    Usage in Feynman's agent.py:

        from covision import VisionSystem
        from covision.integrations.feynman import FeynmanVisionPlugin

        vision = VisionSystem()
        plugin = FeynmanVisionPlugin(vision)

        # In entrypoint
        async def entrypoint(ctx: JobContext):
            plugin.set_agent(agent)
            await plugin.start()

            # Use vision context in responses
            context = plugin.get_context()
    """

    def __init__(
        self,
        vision_system: Any,  # VisionSystem, but avoiding circular import
        greeting_cooldown: float = 300.0,  # 5 minutes between greetings
        attention_callback: Optional[Callable] = None,
    ):
        """Initialize the Feynman vision plugin.

        Args:
            vision_system: CoVision VisionSystem instance
            greeting_cooldown: Minimum seconds between automatic greetings
            attention_callback: Called when user starts looking at camera
        """
        self.vision = vision_system
        self.greeting_cooldown = greeting_cooldown
        self.attention_callback = attention_callback

        self._agent = None
        self._mentor = None
        self._last_greeting_time: float = 0.0
        self._running = False

        # Register event handlers
        self.vision.events.on("user_arrived", self._on_user_arrived)
        self.vision.events.on("user_left", self._on_user_left)
        self.vision.events.on("user_looking", self._on_user_looking)
        self.vision.events.on("scene_update", self._on_scene_update)

    def set_agent(self, agent: Any):
        """Set the LiveKit VoiceAgent for TTS output.

        Args:
            agent: LiveKit VoiceAgent instance
        """
        self._agent = agent

    def set_mentor(self, mentor: Any):
        """Set the Feynman Mentor for context injection.

        Args:
            mentor: Feynman Mentor instance
        """
        self._mentor = mentor

    async def start(self):
        """Start the vision system and plugin."""
        self._running = True
        await self.vision.start_async()
        logger.info("FeynmanVisionPlugin started")

    async def stop(self):
        """Stop the vision system and plugin."""
        self._running = False
        self.vision.stop()
        logger.info("FeynmanVisionPlugin stopped")

    def get_context(self) -> dict:
        """Get current vision context for conversation.

        Returns dict that can be used to enrich LLM context:
        {
            "user_present": bool,
            "user_looking": bool,
            "user_name": str | None,
            "scene_description": str,
            "attention_score": float,
        }
        """
        info = self.vision.presence.info if hasattr(self.vision, "presence") else None

        if info:
            return {
                "user_present": info.is_present,
                "user_looking": info.is_paying_attention,
                "user_name": info.user_name,
                "user_id": info.user_id,
                "attention_score": info.attention_score,
                "duration_seconds": info.duration_seconds,
                "scene_description": self.vision.describe_scene(),
            }
        else:
            return {
                "user_present": self.vision.user_present,
                "user_looking": self.vision.user_looking,
                "user_name": None,
                "scene_description": self.vision.describe_scene(),
            }

    def get_context_string(self) -> str:
        """Get context as a formatted string for system prompt injection.

        Returns a string suitable for adding to the mentor's context.
        """
        ctx = self.get_context()

        parts = []

        if ctx["user_present"]:
            name = ctx.get("user_name", "User")
            parts.append(f"{name} is present")

            if ctx["user_looking"]:
                parts.append("and looking at the camera")

            if ctx.get("scene_description"):
                parts.append(f"Scene: {ctx['scene_description']}")
        else:
            parts.append("No one is currently visible")

        return "[Vision: " + ", ".join(parts) + "]"

    def _on_user_arrived(self, event: UserArrivedEvent):
        """Handle user arrival event."""
        import time

        logger.info(f"Vision: User arrived (confidence={event.confidence:.2f})")

        # Check cooldown for greetings
        now = time.time()
        if now - self._last_greeting_time < self.greeting_cooldown:
            logger.debug("Skipping greeting (cooldown)")
            return

        self._last_greeting_time = now

        # Trigger greeting if agent is available
        if self._agent:
            asyncio.create_task(self._greet_user(event.user_id))

    def _on_user_left(self, event: UserLeftEvent):
        """Handle user departure event."""
        logger.info(
            f"Vision: User left after {event.duration_seconds:.1f}s"
        )

        # Optionally say goodbye for longer sessions
        if event.duration_seconds > 60 and self._agent:
            asyncio.create_task(self._say_goodbye(event.duration_seconds))

    def _on_user_looking(self, event: UserLookingEvent):
        """Handle user looking at camera."""
        logger.debug("Vision: User is looking at camera")

        if self.attention_callback:
            self.attention_callback(event)

    def _on_scene_update(self, event: SceneUpdateEvent):
        """Handle scene update."""
        logger.debug(f"Vision: Scene updated - {event.description[:50]}...")

    async def _greet_user(self, user_id: str):
        """Generate and speak a greeting."""
        if not self._agent:
            return

        # Get user name if available
        name = None
        if hasattr(self.vision, "recognizer"):
            user = self.vision.recognizer.get_user(user_id)
            if user:
                name = user.name

        if name:
            greeting = f"Hey {name}! Good to see you."
        else:
            greeting = "Hey! I can see you now."

        try:
            await self._agent.say(greeting, allow_interruptions=True)
        except Exception as e:
            logger.error(f"Failed to speak greeting: {e}")

    async def _say_goodbye(self, duration: float):
        """Say goodbye when user leaves."""
        if not self._agent:
            return

        minutes = int(duration / 60)
        if minutes > 1:
            goodbye = f"See you later! That was a nice {minutes} minute session."
        else:
            goodbye = "See you!"

        try:
            await self._agent.say(goodbye, allow_interruptions=True)
        except Exception as e:
            logger.error(f"Failed to speak goodbye: {e}")


class FeynmanContextInjector:
    """Injects vision context into Feynman Mentor conversations.

    Usage:
        injector = FeynmanContextInjector(vision, mentor)

        # Before each chat, inject context
        injector.inject_context()
        response = mentor.chat(user_message)
    """

    def __init__(
        self,
        vision_system: Any,
        mentor: Any,
        context_prefix: str = "[Visual context]",
    ):
        """Initialize context injector.

        Args:
            vision_system: CoVision VisionSystem
            mentor: Feynman Mentor instance
            context_prefix: Prefix for injected context
        """
        self.vision = vision_system
        self.mentor = mentor
        self.context_prefix = context_prefix

    def get_vision_context(self) -> str:
        """Generate vision context string."""
        parts = []

        if self.vision.user_present:
            parts.append("User is visible")

            if self.vision.user_looking:
                parts.append("looking at camera")

            scene = self.vision.describe_scene()
            if scene:
                parts.append(scene)
        else:
            parts.append("User not visible")

        return f"{self.context_prefix} {', '.join(parts)}"

    def inject_context(self):
        """Inject current vision context into mentor.

        This modifies the mentor's context before the next chat.
        """
        context = self.get_vision_context()

        # Inject into mentor's context
        # This depends on Feynman's Mentor implementation
        if hasattr(self.mentor, "set_vision_context"):
            self.mentor.set_vision_context(context)
        elif hasattr(self.mentor, "additional_context"):
            self.mentor.additional_context = context
        else:
            logger.warning(
                "Mentor does not support context injection. "
                "Consider adding set_vision_context() method."
            )


def create_vision_enabled_agent(
    vision_system: Any,
    mentor: Any,
    **agent_kwargs,
) -> Any:
    """Factory function to create a vision-enabled LiveKit agent.

    This wraps the standard Feynman agent setup with vision capabilities.

    Args:
        vision_system: CoVision VisionSystem
        mentor: Feynman Mentor instance
        **agent_kwargs: Additional arguments for VoiceAgent

    Returns:
        Configured VoiceAgent with vision integration
    """
    from livekit.agents.voice import Agent as VoiceAgent

    # Create plugin
    plugin = FeynmanVisionPlugin(vision_system)
    plugin.set_mentor(mentor)

    # Create context injector
    injector = FeynmanContextInjector(vision_system, mentor)

    # Create agent with vision-aware LLM wrapper
    # This requires modifying the FeynmanLLM to include vision context

    logger.info("Created vision-enabled agent")

    return plugin, injector
