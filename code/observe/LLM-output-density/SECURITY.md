# Security Policy

Do not commit API keys, Hugging Face tokens, cloud credentials, private model URLs,
or credentials embedded in shell commands. Use interactive login, environment
variables, or a secret manager.

Before pushing any branch, run:

```bash
make verify
git diff --check
```

The original working repository previously contained a Hugging Face token in its
README and must not be published with its existing Git history. This clean repository
starts with new history, but the exposed token must also be revoked in the Hugging
Face account settings.
