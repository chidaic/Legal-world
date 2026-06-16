"""Tools for lawyer or judge style professional roles."""

from .appeal_drafting_tool import (
    APPEAL_DRAFT_TOOL_NAME,
    AppealDraftingTool,
    create_appeal_drafting_tool,
)
from .appeal_response_drafting_tool import (
    APPEAL_RESPONSE_DRAFT_TOOL_NAME,
    AppealResponseDraftingTool,
    create_appeal_response_drafting_tool,
)
from .case_retrieval_tool import (
    CASE_RETRIEVAL_TOOL_NAME,
    LocalCaseRetrievalEngine,
    create_case_retrieval_tool,
    create_case_search_function,
)
from .citation_check_tool import (
    CITATION_CHECK_TOOL_NAME,
    CitationCheckTool,
    create_citation_check_tool,
)
from .benchmark_eval_tool import (
    BENCHMARK_EVAL_TOOL_NAME,
    BenchmarkEvalTool,
    create_benchmark_eval_tool,
)
from .complaint_drafting_tool import (
    COMPLAINT_DRAFT_TOOL_NAME,
    ComplaintDraftingTool,
    create_complaint_drafting_tool,
)
from .document_compare_tool import (
    DOCUMENT_COMPARE_TOOL_NAME,
    DocumentCompareTool,
    create_document_compare_tool,
)
from .defense_drafting_tool import (
    DEFENSE_DRAFT_TOOL_NAME,
    DefenseDraftingTool,
    create_defense_drafting_tool,
)
from .first_instance_judgment_drafting_tool import (
    FIRST_INSTANCE_JUDGMENT_DRAFT_TOOL_NAME,
    FirstInstanceJudgmentDraftingTool,
    create_first_instance_judgment_drafting_tool,
)
from .second_instance_judgment_drafting_tool import (
    SECOND_INSTANCE_JUDGMENT_DRAFT_TOOL_NAME,
    SecondInstanceJudgmentDraftingTool,
    create_second_instance_judgment_drafting_tool,
)
from .document_drafting_registry import (
    create_document_drafting_tool_for_scenario,
    extract_document_drafting_tool_payload,
    get_document_drafting_result_field,
    get_document_drafting_tool_name,
    get_document_type_for_scenario,
    normalize_document_drafting_payload,
    normalize_document_drafting_type,
    render_document_drafting_payload,
    render_document_drafting_payload_for_output_dir,
)
from .judgment_drafting_registry import (
    create_judgment_document_tool_for_scenario,
    extract_judgment_document_tool_payload,
    get_judgment_document_tool_name,
    get_judgment_document_type_for_scenario,
    normalize_judgment_document_payload,
    normalize_judgment_document_type,
    render_judgment_document_payload,
)
from .save_lawyer_memory_tool import create_save_lawyer_memory_tool, normalize_lawyer_memory

__all__ = [
    "APPEAL_DRAFT_TOOL_NAME",
    "APPEAL_RESPONSE_DRAFT_TOOL_NAME",
    "BENCHMARK_EVAL_TOOL_NAME",
    "CASE_RETRIEVAL_TOOL_NAME",
    "CITATION_CHECK_TOOL_NAME",
    "COMPLAINT_DRAFT_TOOL_NAME",
    "DEFENSE_DRAFT_TOOL_NAME",
    "DOCUMENT_COMPARE_TOOL_NAME",
    "FIRST_INSTANCE_JUDGMENT_DRAFT_TOOL_NAME",
    "LocalCaseRetrievalEngine",
    "SECOND_INSTANCE_JUDGMENT_DRAFT_TOOL_NAME",
    "AppealDraftingTool",
    "AppealResponseDraftingTool",
    "BenchmarkEvalTool",
    "CitationCheckTool",
    "ComplaintDraftingTool",
    "DefenseDraftingTool",
    "DocumentCompareTool",
    "FirstInstanceJudgmentDraftingTool",
    "SecondInstanceJudgmentDraftingTool",
    "create_appeal_drafting_tool",
    "create_appeal_response_drafting_tool",
    "create_benchmark_eval_tool",
    "create_case_retrieval_tool",
    "create_case_search_function",
    "create_citation_check_tool",
    "create_complaint_drafting_tool",
    "create_defense_drafting_tool",
    "create_document_drafting_tool_for_scenario",
    "create_document_compare_tool",
    "create_first_instance_judgment_drafting_tool",
    "create_judgment_document_tool_for_scenario",
    "create_second_instance_judgment_drafting_tool",
    "create_save_lawyer_memory_tool",
    "extract_document_drafting_tool_payload",
    "extract_judgment_document_tool_payload",
    "get_document_drafting_result_field",
    "get_document_drafting_tool_name",
    "get_document_type_for_scenario",
    "get_judgment_document_tool_name",
    "get_judgment_document_type_for_scenario",
    "normalize_document_drafting_payload",
    "normalize_document_drafting_type",
    "normalize_judgment_document_payload",
    "normalize_judgment_document_type",
    "normalize_lawyer_memory",
    "render_document_drafting_payload",
    "render_document_drafting_payload_for_output_dir",
    "render_judgment_document_payload",
]
