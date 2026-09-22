"""Shared workflow action sets and phase mapping."""

# Three completed, unsuccessful psychometric repair rounds transition an item
# to the existing defer-confirmation queue.  This is a workflow policy, not a
# model-call retry budget.
PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS = 3

ITEM_DEVELOPMENT_ACTIONS = {
    "generate_item",
    "review_item",
    "revise_item",
    "regenerate_item",
}
DETERMINISTIC_AUTO_APPROVAL_ACTIONS = {
    "simulate_responses",
    "analyze_psychometrics",
    "select_items",
    "confirm_psychometric_repair",
    "psychometric_repair_batch",
    "assemble_test",
    "review_test",
    "rescore_test",
    "generate_reports",
}
PHASE_BY_ACTION = {
    "clarify_requirements": "requirements",
    "build_blueprint": "construct_blueprint",
    "generate_item": "item_development",
    "review_item": "item_development",
    "revise_item": "item_development",
    "regenerate_item": "item_development",
    "simulate_responses": "virtual_simulation",
    "analyze_psychometrics": "psychometric_analysis",
    "select_items": "item_selection",
    "confirm_psychometric_repair": "item_selection",
    "psychometric_repair_batch": "item_selection",
    "assemble_test": "test_assembly",
    "review_test": "test_assembly",
    "rescore_test": "test_assembly",
    "generate_reports": "reporting",
}
