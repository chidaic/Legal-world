"""Data loader for legal case data.

This module provides the DataLoader class for loading and extracting
information from legal case JSON files.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for legal case JSON files.
    
    Loads case data from JSON files and provides methods to extract
    various types of information for different agents and scenarios.
    
    Attributes:
        json_path: Path to the JSON data file
        cases: List of loaded case data dictionaries
    """
    
    def __init__(self, json_path: str):
        """Initialize data loader.
        
        Args:
            json_path: Path to the JSON data file
        """
        self.json_path = json_path
        self.cases: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load JSON data from file."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both list and single object
                if isinstance(data, list):
                    self.cases = data
                else:
                    self.cases = [data]
            logger.info(f"Loaded {len(self.cases)} cases from {self.json_path}")
        except Exception as e:
            logger.error(f"Failed to load data from {self.json_path}: {e}")
            self.cases = []
    
    def get_case(self, case_id: int = 0) -> Dict[str, Any]:
        """Get case data by original_id.

        Args:
            case_id: Case original_id (default: 0)

        Returns:
            Case data dictionary, or empty dict if not found
        """
        # Search by original_id instead of array index
        for case in self.cases:
            if case.get("original_id") == case_id:
                return case

        logger.warning(f"Case with original_id={case_id} not found (total: {len(self.cases)} cases)")
        return {}

    @staticmethod
    def _normalize_name(name: Any) -> str:
        return str(name or "").strip()

    @classmethod
    def _name_matches(cls, expected: Any, actual: Any) -> bool:
        left = cls._normalize_name(expected)
        right = cls._normalize_name(actual)
        if not left or not right:
            return False
        return left == right or left in right or right in left

    def _extract_party_names(self, case: Dict[str, Any], party_role: str) -> List[str]:
        party_info = case.get("extracted_info", {}).get("party_info", {})
        party_raw = party_info.get("plaintiff" if party_role == "plaintiff" else "defendant", {})

        if isinstance(party_raw, list):
            return [
                self._normalize_name(item.get("name", ""))
                for item in party_raw
                if isinstance(item, dict) and self._normalize_name(item.get("name", ""))
            ]

        if isinstance(party_raw, dict):
            name = self._normalize_name(party_raw.get("name", ""))
            return [name] if name else []

        return []

    def _case_matches_profile(self, case: Dict[str, Any], party_role: str, profile_name: str) -> bool:
        normalized_profile_name = self._normalize_name(profile_name)
        if not case or not normalized_profile_name:
            return False

        return any(
            self._name_matches(normalized_profile_name, candidate_name)
            for candidate_name in self._extract_party_names(case, party_role)
        )

    @classmethod
    def _merge_mapping_list(cls, items: List[Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if key not in merged or merged[key] in ("", None, [], {}):
                    merged[key] = value
                elif isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = {**merged[key], **value}
                elif isinstance(merged[key], list) and isinstance(value, list):
                    merged[key] = [*merged[key], *value]
        return merged

    @classmethod
    def _as_mapping(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return cls._merge_mapping_list(value)
        return {}

    @classmethod
    def _extract_instance_mapping(cls, case: Dict[str, Any], instance_key: str) -> Dict[str, Any]:
        extracted_info = case.get("extracted_info", {})
        if not isinstance(extracted_info, dict):
            return {}
        return cls._as_mapping(extracted_info.get(instance_key, {}))

    @classmethod
    def _extract_text(cls, value: Any, *keys: str) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in keys:
                text = cls._extract_text(value.get(key))
                if text:
                    return text
            return ""
        if isinstance(value, list):
            parts = [cls._extract_text(item, *keys) for item in value]
            return "\n".join(part for part in parts if part)
        return str(value).strip() if value else ""

    def resolve_case(
        self,
        *,
        case_id: int | str | None = 0,
        party_role: str = "plaintiff",
        profile_name: str = "",
    ) -> Dict[str, Any]:
        """Resolve a case robustly for sandbox agents.

        The sandbox directory often uses stable local case folders like ``case_1``,
        while the copied dataset uses ``original_id`` values that may not align with
        those folder numbers. We therefore try the declared ``case_id`` first, then
        fall back to matching by the current party's profile name.
        """
        candidate: Dict[str, Any] = {}
        normalized_case_id = str(case_id or "").strip()
        if normalized_case_id.startswith("case_"):
            normalized_case_id = normalized_case_id[5:]

        if normalized_case_id:
            try:
                target_case_id = int(normalized_case_id)
                candidate = next(
                    (case for case in self.cases if case.get("original_id") == target_case_id),
                    {},
                )
            except (TypeError, ValueError):
                logger.warning("Invalid case_id for case resolution: %s", case_id)

        if candidate and self._case_matches_profile(candidate, party_role, profile_name):
            return candidate

        normalized_profile_name = self._normalize_name(profile_name)
        if normalized_profile_name:
            for case in self.cases:
                if self._case_matches_profile(case, party_role, normalized_profile_name):
                    resolved_case_id = case.get("original_id", "")
                    if candidate:
                        logger.warning(
                            "Sandbox case_id=%s mismatched dataset original_id=%s; "
                            "resolved by %s profile name=%s",
                            case_id,
                            candidate.get("original_id", ""),
                            party_role,
                            normalized_profile_name,
                        )
                    else:
                        logger.info(
                            "Resolved dataset case by %s profile name=%s -> original_id=%s",
                            party_role,
                            normalized_profile_name,
                            resolved_case_id,
                        )
                    return case

            if candidate:
                logger.warning(
                    "Sandbox case_id=%s resolved to dataset original_id=%s, but %s profile name=%s "
                    "did not match any dataset case; returning empty case to avoid cross-case contamination",
                    case_id,
                    candidate.get("original_id", ""),
                    party_role,
                    normalized_profile_name,
                )
                return {}

        return candidate

    def resolve_case_for_config(
        self,
        config: Dict[str, Any],
        fallback_name: str = "",
        fallback_dataset_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        profile = config.get("profile", {})
        profile_name = ""
        if isinstance(profile, dict):
            profile_name = self._normalize_name(profile.get("name", ""))
        if not profile_name:
            profile_name = self._normalize_name(fallback_name)

        resolved_case = self.resolve_case(
            case_id=config.get("case_id", "0"),
            party_role=str(config.get("party_role", "plaintiff") or "plaintiff"),
            profile_name=profile_name,
        )
        if resolved_case or not fallback_dataset_paths:
            return resolved_case

        party_role = str(config.get("party_role", "plaintiff") or "plaintiff")
        case_id = config.get("case_id", "0")
        normalized_current_path = str(Path(self.json_path).resolve()) if self.json_path else ""
        for candidate_path in fallback_dataset_paths:
            candidate = str(candidate_path or "").strip()
            if not candidate:
                continue

            candidate_resolved = str(Path(candidate).resolve())
            if normalized_current_path and candidate_resolved == normalized_current_path:
                continue

            fallback_loader = DataLoader(candidate)
            fallback_case = fallback_loader.resolve_case(
                case_id=case_id,
                party_role=party_role,
                profile_name=profile_name,
            )
            if fallback_case:
                logger.info(
                    "Resolved dataset case via fallback path %s for %s profile name=%s",
                    candidate,
                    party_role,
                    profile_name,
                )
                return fallback_case

        return {}
    
    def get_case_count(self) -> int:
        """Get total number of cases.
        
        Returns:
            Number of cases loaded
        """
        return len(self.cases)

    @staticmethod
    def _extract_legal_persona_profile(profile: Any) -> Dict[str, str]:
        if not isinstance(profile, dict):
            return {}

        extracted: Dict[str, str] = {}
        for field in (
            "legal_literacy_level",
            "information_disclosure_willingness",
            "emotional_stability",
            "narrative_proficiency",
        ):
            value = str(profile.get(field, "") or "").strip().lower()
            if value in {"high", "medium", "low"}:
                extracted[field] = value
        return extracted

    @staticmethod
    def _is_corporate_party(party_type: Any) -> bool:
        normalized = str(party_type or "").strip().lower()
        return normalized in {"法人", "企业", "公司", "corporate", "company", "legal_person"}

    @classmethod
    def normalize_party_profile(cls, profile: Any) -> Dict[str, Any]:
        if not isinstance(profile, dict):
            return {}

        normalized = dict(profile)
        party_type = str(
            normalized.get("party_type", "")
            or normalized.get("type", "")
            or ""
        ).strip()
        normalized["type"] = party_type
        normalized["party_type"] = party_type

        if cls._is_corporate_party(party_type):
            normalized["gender"] = ""
            normalized["ethnicity"] = ""
            normalized["birth_date"] = ""

        return normalized
    
    def extract_plaintiff_profile(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract plaintiff profile (固有字段) from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Plaintiff profile dictionary
        """
        party_info = case.get("extracted_info", {}).get("party_info", {})
        plaintiff = self.normalize_party_profile(party_info.get("plaintiff", {}))
        
        # Extract only the question text, excluding reference answers
        questions_raw = plaintiff.get("questions", [])
        questions = []
        for q in questions_raw:
            if isinstance(q, dict) and "question" in q:
                questions.append(q["question"])
            elif isinstance(q, str):
                questions.append(q)
        
        return {
            "name": plaintiff.get("name", ""),
            "type": plaintiff.get("type", ""),
            "party_type": plaintiff.get("type", ""),
            "gender": plaintiff.get("gender", ""),
            "ethnicity": plaintiff.get("ethnicity", ""),
            "birth_date": plaintiff.get("birth_date", ""),
            "address": plaintiff.get("address", ""),
            "personality": plaintiff.get("personality", ""),
            "speaking_style": plaintiff.get("speaking_style", ""),
            "interaction_guidelines": plaintiff.get("interaction_guidelines", ""),
            "representative": plaintiff.get("representative", ""),
            "legal_persona_profile": self._extract_legal_persona_profile(
                plaintiff.get("legal_persona_profile", {})
            ),
            "questions": questions,
        }
    
    def extract_defendant_profile(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract defendant profile (固有字段) from case data.

        Handles both single defendant (dict) and multiple defendants (list).
        When defendant is a list, uses the first defendant as the primary profile
        and provides all defendant names.

        Args:
            case: Case data dictionary

        Returns:
            Defendant profile dictionary
        """
        party_info = case.get("extracted_info", {}).get("party_info", {})
        defendant_raw = party_info.get("defendant", {})

        # Handle defendant being a list (multiple defendants)
        if isinstance(defendant_raw, list) and len(defendant_raw) > 0:
            defendant = self.normalize_party_profile(defendant_raw[0])  # Primary defendant
            all_names = "、".join(d.get("name", "") for d in defendant_raw)
        else:
            defendant = self.normalize_party_profile(defendant_raw if isinstance(defendant_raw, dict) else {})
            all_names = defendant.get("name", "")

        # Extract questions (same as plaintiff)
        questions_raw = defendant.get("questions", [])
        questions = []
        for q in questions_raw:
            if isinstance(q, dict) and "question" in q:
                questions.append(q["question"])
            elif isinstance(q, str):
                questions.append(q)

        return {
            "name": defendant.get("name", ""),
            "all_defendant_names": all_names,
            "type": defendant.get("type", ""),
            "party_type": defendant.get("type", ""),
            "gender": defendant.get("gender", ""),
            "ethnicity": defendant.get("ethnicity", ""),
            "birth_date": defendant.get("birth_date", ""),
            "address": defendant.get("address", ""),
            "personality": defendant.get("personality", ""),
            "speaking_style": defendant.get("speaking_style", ""),
            "interaction_guidelines": defendant.get("interaction_guidelines", ""),
            "representative": defendant.get("representative", ""),
            "legal_persona_profile": self._extract_legal_persona_profile(
                defendant.get("legal_persona_profile", {})
            ),
            "questions": questions,
        }
    
    def extract_case_background(self, case: Dict[str, Any]) -> str:
        """Extract case background from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Case background string
        """
        return case.get("extracted_info", {}).get("case_background", "")
    
    def extract_case_id(self, case: Dict[str, Any]) -> str:
        """Extract case ID from case data.
        
        Uses original_id field from the new dataset format.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Case ID string
        """
        return str(case.get("original_id", ""))
    
    def extract_case_cause(self, case: Dict[str, Any]) -> str:
        """Extract case cause (案由) from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Case cause string (e.g. '买卖合同纠纷')
        """
        return case.get("extracted_info", {}).get("case_cause", "")
    
    def extract_case_number(self, case: Dict[str, Any], instance: str = "first") -> str:
        """Extract case number (案号) from case data.
        
        Args:
            case: Case data dictionary
            instance: 'first' for first instance, 'second' for second instance
            
        Returns:
            Case number string (e.g. '（2018）豫1602民初6320号')
        """
        instance_key = "first_instance" if instance == "first" else "second_instance"
        return self._extract_instance_mapping(case, instance_key).get("case_number", "")
    
    def extract_judge_name(self, case: Dict[str, Any], instance: str = "first") -> str:
        """Extract judge name (审判员姓名) from case data.
        
        Args:
            case: Case data dictionary
            instance: 'first' for first instance, 'second' for second instance
            
        Returns:
            First judge's name string
        """
        instance_key = "first_instance" if instance == "first" else "second_instance"
        judges = self._extract_instance_mapping(case, instance_key).get("judges", [])
        return judges[0] if judges else ""
    
    def extract_court_name(self, case: Dict[str, Any], instance: str = "first") -> str:
        """Extract court name (法院名称) from case data.
        
        Args:
            case: Case data dictionary
            instance: 'first' for first instance, 'second' for second instance
            
        Returns:
            Court name string (e.g. '河南省周口市川汇区人民法院')
        """
        instance_key = "first_instance" if instance == "first" else "second_instance"
        return self._extract_instance_mapping(case, instance_key).get("court", "")
    
    def extract_first_instance_info(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract first instance information from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            First instance info dictionary
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        
        # Extract plaintiff_claim
        plaintiff_claim = first_instance.get("plaintiff_claim", {})
        claims = plaintiff_claim.get("claim", [])
        
        # Extract defendant_plea
        defendant_plea = first_instance.get("defendant_plea", {})
        defense = self._extract_text(defendant_plea, "plea", "defense")
        
        # Extract judgment result
        final_judgment = first_instance.get("final_judgment", {})
        judgment_result = final_judgment.get("judgment_result", [])
        
        return {
            "claims": claims,
            "facts_and_reasons": first_instance.get("facts_and_reasons", ""),
            "defendant_defense": defense,
            "court_finding": first_instance.get("court_finding", ""),
            "court_opinion": first_instance.get("court_opinion", ""),
            "judgment_result": judgment_result,
            "evidence": first_instance.get("evidence", {}),
            "judges": first_instance.get("judges", []),
            "court": first_instance.get("court", ""),
            "case_number": first_instance.get("case_number", ""),
        }
    
    def extract_second_instance_info(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract second instance information from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Second instance info dictionary
        """
        second_instance = self._extract_instance_mapping(case, "second_instance")
        appellant_claim_raw = second_instance.get("appellant_claim", {})
        appellant_claim = self._as_mapping(appellant_claim_raw)
        appellee_defense_raw = second_instance.get("appellee_defense", {})
        court_finding = second_instance.get("court_finding", "") or second_instance.get("court_findings", "")
        court_opinion = second_instance.get("court_opinion", "") or second_instance.get("judgment", "")
        respondent_defense = (
            second_instance.get("respondent_defense", "")
            or self._extract_text(appellee_defense_raw, "defense", "plea")
        )

        return {
            "appeal_claims": second_instance.get("appeal_claims", "") or appellant_claim.get("claim", []),
            "appeal_reasons": (
                second_instance.get("appeal_reasons", "")
                or self._extract_text(appellant_claim_raw, "reasons")
                or appellant_claim.get("reasons", "")
            ),
            "respondent_defense": respondent_defense,
            "court_finding": court_finding,
            "court_findings": court_finding,
            "court_opinion": court_opinion,
            "judgment": court_opinion,
        }
    
    def extract_claims(self, case: Dict[str, Any]) -> str:
        """Extract claims (诉讼请求) from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Claims string (formatted as numbered list if array)
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        plaintiff_claim = self._as_mapping(first_instance.get("plaintiff_claim", {}))
        claims = plaintiff_claim.get("claim", [])
        
        if isinstance(claims, list):
            return "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        return str(claims) if claims else ""
    
    def extract_evidence(self, case: Dict[str, Any]) -> str:
        """Extract evidence from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Evidence string (formatted from nested structure)
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        evidence = first_instance.get("evidence", {})
        
        # Handle nested evidence structure
        evidence_items = []
        
        # Extract plaintiff evidence
        evidence_mapping = self._as_mapping(evidence)
        plaintiff_evidence = self._as_mapping(evidence_mapping.get("plaintiff_evidence", {}))
        for key, value in plaintiff_evidence.items():
            if isinstance(value, dict) and "evidence" in value:
                evidence_items.append(f"原告证据：{value['evidence']}")
        
        # Extract defendant evidence  
        defendant_evidence = self._as_mapping(evidence_mapping.get("defendant_evidence", {}))
        for key, value in defendant_evidence.items():
            if isinstance(value, dict) and "evidence" in value:
                evidence_items.append(f"被告证据：{value['evidence']}")
        
        if evidence_items:
            return "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidence_items))
        
        # Fallback: if evidence is a simple list or string
        if isinstance(evidence, list):
            return "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidence))
        return str(evidence) if evidence else ""
    
    def extract_plaintiff_evidence(self, case: Dict[str, Any]) -> str:
        """仅提取原告证据。
        
        Args:
            case: Case data dictionary
            
        Returns:
            原告证据字符串
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        evidence = self._as_mapping(first_instance.get("evidence", {}))
        plaintiff_evidence = self._as_mapping(evidence.get("plaintiff_evidence", {}))
        
        evidence_items = []
        for key, value in plaintiff_evidence.items():
            if isinstance(value, dict) and "evidence" in value:
                evidence_items.append(value['evidence'])
        
        if evidence_items:
            return "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidence_items))
        return ""
    
    def extract_defendant_evidence(self, case: Dict[str, Any]) -> str:
        """仅提取被告证据。
        
        Args:
            case: Case data dictionary
            
        Returns:
            被告证据字符串
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        evidence = self._as_mapping(first_instance.get("evidence", {}))
        defendant_evidence = self._as_mapping(evidence.get("defendant_evidence", {}))
        
        evidence_items = []
        for key, value in defendant_evidence.items():
            if isinstance(value, dict) and "evidence" in value:
                evidence_items.append(value['evidence'])
        
        if evidence_items:
            return "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidence_items))
        return ""
    
    def extract_facts_and_reasons(self, case: Dict[str, Any]) -> str:
        """Extract facts and reasons from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Facts and reasons string
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        return first_instance.get("facts_and_reasons", "")
    
    def extract_defendant_defense(self, case: Dict[str, Any]) -> str:
        """Extract defendant's defense from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Defendant defense string
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        # Prioritize defendant_plea.plea
        plea = self._extract_text(first_instance.get("defendant_plea", {}), "plea", "defense")
        if plea:
            return plea
        return self._extract_text(first_instance.get("defendant_defense", ""))

    def extract_court_opinion(self, case: Dict[str, Any]) -> str:
        """Extract court opinion (本院认为) from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Court opinion string
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        return first_instance.get("court_opinion", "")

    def extract_second_instance_court_opinion(self, case: Dict[str, Any]) -> str:
        """Extract second-instance court opinion with backward-compatible fallbacks."""
        second_instance = self._extract_instance_mapping(case, "second_instance")
        return second_instance.get("court_opinion", "") or second_instance.get("judgment", "")

    def extract_second_instance_appellee_defense(self, case: Dict[str, Any]) -> str:
        """Extract appellee defense from second-instance data."""
        second_instance = self._extract_instance_mapping(case, "second_instance")
        appellee_defense = second_instance.get("appellee_defense", {})
        return self._extract_text(appellee_defense, "defense", "plea")

    def extract_evidence_disputes(self, case: Dict[str, Any], side: str = "plaintiff") -> str:
        """Extract disputes regarding a specific side's evidence.
        
        Args:
            case: Case data dictionary
            side: "plaintiff" or "defendant" (whose evidence disputes to extract)
            
        Returns:
            Formatted string of evidence disputes
        """
        first_instance = self._extract_instance_mapping(case, "first_instance")
        evidence = self._as_mapping(first_instance.get("evidence", {}))
        
        target_key = "plaintiff_evidence" if side == "plaintiff" else "defendant_evidence"
        target_evidence = self._as_mapping(evidence.get(target_key, {}))
        
        dispute_items = []
        for key, value in target_evidence.items():
            if isinstance(value, dict):
                content = value.get("evidence", "")
                dispute = value.get("dispute", "")
                if dispute:
                    dispute_items.append(f"针对证据【{content}】的质证意见：{dispute}")
        
        if dispute_items:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(dispute_items))
        return "暂无明确质证意见"

    def extract_second_instance_evidence(self, case: Dict[str, Any], side: str = "appellant") -> str:
        """Extract second-instance new evidence for the given side."""
        second_instance = self._extract_instance_mapping(case, "second_instance")
        new_evidence = self._as_mapping(second_instance.get("new_evidence", {}))
        target_key = "appellant_evidence" if side == "appellant" else "appellee_evidence"
        target_evidence = self._as_mapping(new_evidence.get(target_key, {}))

        evidence_items = []
        for _, value in target_evidence.items():
            if isinstance(value, dict) and "evidence" in value:
                evidence_items.append(value["evidence"])

        if evidence_items:
            return "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidence_items))
        return ""

    def extract_second_instance_evidence_disputes(self, case: Dict[str, Any], side: str = "appellant") -> str:
        """Extract disputes for second-instance new evidence."""
        second_instance = self._extract_instance_mapping(case, "second_instance")
        new_evidence = self._as_mapping(second_instance.get("new_evidence", {}))
        target_key = "appellant_evidence" if side == "appellant" else "appellee_evidence"
        target_evidence = self._as_mapping(new_evidence.get(target_key, {}))

        dispute_items = []
        for _, value in target_evidence.items():
            if isinstance(value, dict):
                content = value.get("evidence", "")
                dispute = value.get("dispute", "")
                if dispute:
                    dispute_items.append(f"针对证据【{content}】的质证意见：{dispute}")

        if dispute_items:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(dispute_items))
        return "暂无明确质证意见"

    def extract_all_second_instance_evidence(self, case: Dict[str, Any]) -> str:
        """Extract all second-instance new evidence."""
        second_instance = self._extract_instance_mapping(case, "second_instance")
        new_evidence = self._as_mapping(second_instance.get("new_evidence", {}))

        evidence_items = []
        for target_key, side_label in (
            ("appellant_evidence", "上诉人"),
            ("appellee_evidence", "被上诉人"),
        ):
            target_evidence = self._as_mapping(new_evidence.get(target_key, {}))
            for _, value in target_evidence.items():
                if isinstance(value, dict) and "evidence" in value:
                    evidence_items.append(f"{side_label}证据：{value['evidence']}")

        if evidence_items:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(evidence_items))
        return ""

    def extract_all_second_instance_evidence_disputes(self, case: Dict[str, Any]) -> str:
        """Extract disputes for all second-instance new evidence."""
        second_instance = self._extract_instance_mapping(case, "second_instance")
        new_evidence = self._as_mapping(second_instance.get("new_evidence", {}))

        dispute_items = []
        for target_key, side_label in (
            ("appellant_evidence", "上诉人"),
            ("appellee_evidence", "被上诉人"),
        ):
            target_evidence = self._as_mapping(new_evidence.get(target_key, {}))
            for _, value in target_evidence.items():
                if isinstance(value, dict):
                    content = value.get("evidence", "")
                    dispute = value.get("dispute", "")
                    if dispute:
                        dispute_items.append(f"{side_label}方证据【{content}】的质证意见：{dispute}")

        if dispute_items:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(dispute_items))
        return "暂无明确质证意见"

    def extract_all_evidence_disputes(self, case: Dict[str, Any]) -> str:
        """Extract disputes for all first-instance evidence."""
        first_instance = self._extract_instance_mapping(case, "first_instance")
        evidence = self._as_mapping(first_instance.get("evidence", {}))

        dispute_items = []
        for target_key, side_label in (
            ("plaintiff_evidence", "原告"),
            ("defendant_evidence", "被告"),
        ):
            target_evidence = self._as_mapping(evidence.get(target_key, {}))
            for _, value in target_evidence.items():
                if isinstance(value, dict):
                    content = value.get("evidence", "")
                    dispute = value.get("dispute", "")
                    if dispute:
                        dispute_items.append(f"{side_label}方证据【{content}】的质证意见：{dispute}")

        if dispute_items:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(dispute_items))
        return "暂无明确质证意见"

    def extract_appellant_appeal(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract appellant appeal info (claim and reasons) from case data.
        
        Args:
            case: Case data dictionary
            
        Returns:
            Dictionary containing 'claim' (list) and 'reasons' (str)
        """
        second_instance = self._extract_instance_mapping(case, "second_instance")
        appellant_claim = self._as_mapping(second_instance.get("appellant_claim", {}))
        
        return {
            "claim": appellant_claim.get("claim", []),
            "reasons": appellant_claim.get("reasons", ""),
        }
