"""Program-owned PSJT format constants.

These are implementation invariants, not user requirements and not model
output fields.
"""

PSJT_OPTION_COUNT = 4
PSJT_RESPONSE_INSTRUCTION = "你会怎么做？"
PSJT_SCORING_METHOD = "1-4行为高低计分"
DEFAULT_OUTPUT_LANGUAGE = "zh-CN"
DEFAULT_USER_REQUEST = (
    "请帮我出一个覆盖 NEO-PI-R 全部 facet、每个 facet 5 道题的人格情境判断测验，"
    "测量大学生，用于人格测评"
)
DEFAULT_TARGET_POPULATION = "大学生"
DEFAULT_TARGET_CONSTRUCT = "NEO-PI-R 全部 facet"
DEFAULT_REQUESTED_ITEM_COUNT = 150

