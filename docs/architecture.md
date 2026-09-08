# Architecture

The application begins as a single Android app module to keep the initial codebase approachable.

## Layers

- **UI:** Activities, fragments, and XML layouts render user-facing screens.
- **Domain:** Future use cases and model types express application behavior independently of Android framework classes.
- **Data:** Future repositories provide access to local persistence and remote services.

As the application grows, dependencies should point inward: UI depends on domain contracts, while data implements those contracts.

## Initial structure

```text
app/
  src/
    main/
      java/com/example/remember/
      res/
docs/
```
