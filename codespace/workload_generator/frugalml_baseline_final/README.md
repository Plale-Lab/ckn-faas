# Standalone FrugalML Baseline Template

This is a template project structure for integrating a standalone
FrugalML-style baseline with your existing Iluvatar deployment.

Files included:
- config.py
- policy.py
- iluvatar_client.py
- run_workload.py
- train_policy.py

NOTE:
The code snippets from our discussion reference functions from your
existing DYSCI/Iluvatar project (e.g., build_stub_map_for_selected_models,
protobuf stubs, wnid_matches_text_label). Those project-specific functions
must be imported from your existing codebase.
