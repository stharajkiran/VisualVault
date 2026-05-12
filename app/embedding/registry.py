from app.embedding.provider import EmbeddingProvider


class ModelRegistry:
    """
    Per-process registry of named EmbeddingProvider instances.

    Holds references to already-loaded providers and tracks which one is active.
    set_active() is a startup-time decision — changing it after workers are running
    only affects the calling process, not other Celery workers.
    """

    def __init__(self) -> None:
        self._providers: dict[str, EmbeddingProvider] = {}
        self._active: str | None = None

    def register(self, provider: EmbeddingProvider) -> None:
        """
        Add a provider to the registry under its config alias.

        The alias comes from provider.config.alias (e.g. "clip-b32", "clip-l14").
        Registering a second provider with the same alias overwrites the first.
        If no active provider is set yet, the first registered provider becomes active.

        Args:
            provider (EmbeddingProvider): A loaded provider instance to register.

        Returns:
            None
        """
        alias = provider.config.alias
        self._providers[alias] = provider
        if self._active is None:
            self._active = alias

    def get(self, alias: str) -> EmbeddingProvider:
        """
        Retrieve a provider by its alias.

        Args:
            alias (str): The alias defined in CLIPConfig (e.g. "clip-b32").

        Returns:
            EmbeddingProvider: The registered provider for that alias.

        Raises:
            KeyError: If no provider with that alias has been registered.
        """
        if alias not in self._providers:
            raise KeyError(f"No provider registered under alias '{alias}'. "
                           f"Registered: {list(self._providers.keys())}")
        return self._providers[alias]

    def get_active(self) -> EmbeddingProvider:
        """
        Return the currently active provider.

        Returns:
            EmbeddingProvider: The active provider instance.

        Raises:
            RuntimeError: If no provider has been registered yet.
        """
        if self._active is None:
            raise RuntimeError("No providers registered. Call register() before get_active().")
        return self._providers[self._active]

    def set_active(self, alias: str) -> None:
        """
        Set the active provider by alias.

        This is a startup-time decision. In a multi-worker Celery environment,
        calling this after workers have started only affects the calling process.

        Args:
            alias (str): Alias of a previously registered provider.

        Returns:
            None

        Raises:
            KeyError: If the alias has not been registered.
        """
        if alias not in self._providers:
            raise KeyError(f"Cannot set active: '{alias}' not registered. "
                           f"Registered: {list(self._providers.keys())}")
        self._active = alias

    @property
    def active_alias(self) -> str | None:
        """Return the alias of the currently active provider, or None if empty."""
        return self._active

    @property
    def registered_aliases(self) -> list[str]:
        """Return all registered aliases."""
        return list(self._providers.keys())


# Module-level singleton — imported and used directly by FastAPI lifespan and Celery tasks
registry = ModelRegistry()
