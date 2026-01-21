"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Structured token position system for entity binding tasks.

This module provides a template-aware approach to identifying token positions
by parsing the prompt structure using knowledge of the templates and delimiters.

Unlike the naive substring matching in token_positions.py, this system:
- Understands the structure of statements vs questions
- Can distinguish between multiple occurrences of the same entity
- Provides rich metadata about each segment of the prompt
- Correctly identifies which occurrence of an entity is in which region
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re
from .config import EntityBindingTaskConfig, BindingMatrix, EntityGroup
from .templates import TemplateProcessor
from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.pipeline import LMPipeline


@dataclass
class PromptSegment:
    """A single segment of a parsed prompt.

    Represents one piece of the prompt (template text, entity, delimiter, etc.)
    with its position and metadata.
    """
    segment_type: str  # 'template_text', 'delimiter', 'entity', 'question_text'
    text: str  # The actual text content
    char_start: int  # Starting character position in full prompt
    char_end: int  # Ending character position (exclusive)

    # For entity segments only:
    group_idx: Optional[int] = None
    entity_idx: Optional[int] = None

    # For delimiter segments only:
    delimiter_idx: Optional[int] = None

    # Token positions (computed after tokenization):
    token_positions: Optional[List[int]] = None

    def __repr__(self):
        tokens_str = f"tokens={self.token_positions}" if self.token_positions else "no tokens"
        if self.segment_type == 'entity':
            return f"PromptSegment(entity g{self.group_idx}_e{self.entity_idx}: '{self.text}' @ {self.char_start}-{self.char_end}, {tokens_str})"
        else:
            return f"PromptSegment({self.segment_type}: '{self.text}' @ {self.char_start}-{self.char_end}, {tokens_str})"


@dataclass
class ParsedPrompt:
    """Structured representation of a complete prompt.

    Contains the full prompt text broken down into segments, with regions marked.
    """
    raw_text: str
    segments: List[PromptSegment] = field(default_factory=list)
    statement_region: Optional[Tuple[int, int]] = None  # (char_start, char_end)
    question_region: Optional[Tuple[int, int]] = None   # (char_start, char_end)

    def get_entity_segments(self, group_idx: int, entity_idx: int,
                           region: Optional[str] = None) -> List[PromptSegment]:
        """Get all entity segments matching the criteria.

        Parameters
        ----------
        group_idx : int
            Entity group index
        entity_idx : int
            Entity index within group
        region : str, optional
            Which region to search: 'statement', 'question', or None for all

        Returns
        -------
        List[PromptSegment]
            All matching entity segments (may be empty)
        """
        matches = []
        for seg in self.segments:
            if seg.segment_type != 'entity':
                continue
            if seg.group_idx != group_idx or seg.entity_idx != entity_idx:
                continue

            # Check region constraint
            if region == 'statement':
                if not self.statement_region:
                    continue  # No statement region, skip
                if not (self.statement_region[0] <= seg.char_start < self.statement_region[1]):
                    continue
            elif region == 'question':
                if not self.question_region:
                    continue  # No question region, skip
                if not (self.question_region[0] <= seg.char_start < self.question_region[1]):
                    continue

            matches.append(seg)

        return matches

    def get_entity_tokens(self, group_idx: int, entity_idx: int,
                         region: Optional[str] = None) -> List[int]:
        """Get token positions for an entity.

        Parameters
        ----------
        group_idx : int
            Entity group index
        entity_idx : int
            Entity index within group
        region : str, optional
            Which region: 'statement', 'question', or None for first occurrence

        Returns
        -------
        List[int]
            Token position indices

        Raises
        ------
        ValueError
            If entity not found in specified region
        """
        segments = self.get_entity_segments(group_idx, entity_idx, region)

        if not segments:
            region_str = f" in {region} region" if region else ""
            raise ValueError(
                f"Entity g{group_idx}_e{entity_idx} not found{region_str}"
            )

        # Return first match's tokens
        seg = segments[0]
        if seg.token_positions is None:
            raise ValueError(
                f"Entity g{group_idx}_e{entity_idx} has no token positions computed"
            )

        return seg.token_positions

    def __repr__(self):
        return f"ParsedPrompt('{self.raw_text[:50]}...', {len(self.segments)} segments)"


class PromptParser:
    """Parse prompts using template structure knowledge."""

    def __init__(self, config: EntityBindingTaskConfig):
        self.config = config
        self.template_processor = TemplateProcessor(config)

    def parse_prompt(self, input_sample: dict) -> ParsedPrompt:
        """Parse a prompt into structured segments.

        Parameters
        ----------
        input_sample : dict
            Input sample containing entity values and raw_input

        Returns
        -------
        ParsedPrompt
            Structured representation of the prompt
        """
        raw_text = input_sample.get('raw_input')
        if not raw_text:
            raise ValueError("input_sample must contain 'raw_input'")

        # Reconstruct the binding matrix from input_sample
        binding_matrix = self._extract_binding_matrix(input_sample)

        # Parse statement portion
        statement_segments, statement_text = self._parse_statement(binding_matrix)

        # Parse question portion if present
        question_segments = []
        question_text = ""

        if 'query_group' in input_sample and 'query_indices' in input_sample:
            query_group = input_sample['query_group']
            query_indices = input_sample['query_indices']
            answer_index = input_sample.get('answer_index', 0)

            question_segments, question_text = self._parse_question(
                binding_matrix, query_group, query_indices, answer_index
            )

        # Adjust question segment offsets to full prompt coordinates
        if question_text:
            q_start = len(statement_text) + 1  # +1 for space
            for seg in question_segments:
                seg.char_start += q_start
                seg.char_end += q_start

        # Combine segments and mark regions
        all_segments = statement_segments + question_segments

        statement_region = (0, len(statement_text)) if statement_text else None
        question_region = None
        if question_text:
            # Question starts after statement + space separator
            q_start = len(statement_text) + 1  # +1 for space
            question_region = (q_start, q_start + len(question_text))

        return ParsedPrompt(
            raw_text=raw_text,
            segments=all_segments,
            statement_region=statement_region,
            question_region=question_region
        )

    def _extract_binding_matrix(self, input_sample: dict) -> BindingMatrix:
        """Extract binding matrix from input_sample."""
        groups = []
        active_groups = input_sample.get('active_groups', self.config.max_groups)
        entities_per_group = input_sample.get('entities_per_group', self.config.max_entities_per_group)

        for g in range(active_groups):
            entities = []
            for e in range(entities_per_group):
                entity_key = f"entity_g{g}_e{e}"
                entity = input_sample.get(entity_key)
                if entity is not None:
                    entities.append(entity)
                else:
                    # Use placeholder for inactive entities
                    entities.append(None)
            groups.append(EntityGroup(entities, g))

        return BindingMatrix(groups, self.config.max_groups, entities_per_group)

    def _parse_statement(self, binding_matrix: BindingMatrix) -> Tuple[List[PromptSegment], str]:
        """Parse the statement portion of the prompt.

        Returns
        -------
        Tuple[List[PromptSegment], str]
            (segments, full_statement_text)
        """
        # Generate filled statements for each group
        filled_statements = []
        statement_segments_list = []

        for g in range(binding_matrix.get_active_groups()):
            # Fill template for this group
            entity_dict = {
                f"entity_e{i}": binding_matrix.get_entity(g, i)
                for i in range(binding_matrix.get_entities_per_group())
            }

            filled = self.config.statement_template.format(**entity_dict)
            filled_statements.append(filled)

            # Parse this filled statement into segments
            segs = self._parse_filled_template(
                self.config.statement_template,
                entity_dict,
                group_idx=g,
                char_offset=0  # Will adjust later
            )
            statement_segments_list.append(segs)

        # Combine statements with delimiters using statement_conjunction_function
        from causalab.causal.causal_utils import statement_conjunction_function
        full_statement = statement_conjunction_function(filled_statements, self.config.delimiters)

        # Now parse the full statement with proper character offsets
        all_segments = self._parse_statement_with_delimiters(
            filled_statements, full_statement, binding_matrix
        )

        return all_segments, full_statement

    def _parse_filled_template(self, template: str, entity_dict: Dict[str, str],
                               group_idx: int, char_offset: int) -> List[PromptSegment]:
        """Parse a filled template into segments.

        Parameters
        ----------
        template : str
            Template string with {entity_e0}, {entity_e1} placeholders
        entity_dict : Dict[str, str]
            Dictionary mapping placeholder names to entity values
        group_idx : int
            Group index these entities belong to
        char_offset : int
            Character offset to add to all positions

        Returns
        -------
        List[PromptSegment]
            Segments for this template
        """
        segments = []

        # Find all {entity_eN} placeholders
        pattern = r'\{entity_e(\d+)\}'

        current_pos = 0
        for match in re.finditer(pattern, template):
            # Add template text before this entity
            if match.start() > current_pos:
                template_text = template[current_pos:match.start()]
                segments.append(PromptSegment(
                    segment_type='template_text',
                    text=template_text,
                    char_start=char_offset + current_pos,
                    char_end=char_offset + match.start()
                ))

            # Add entity segment
            entity_idx = int(match.group(1))
            placeholder = match.group(0)
            entity_value = entity_dict.get(placeholder[1:-1])  # Remove {}

            if entity_value:
                segments.append(PromptSegment(
                    segment_type='entity',
                    text=entity_value,
                    char_start=char_offset + match.start(),
                    char_end=char_offset + match.start() + len(entity_value),
                    group_idx=group_idx,
                    entity_idx=entity_idx
                ))

            current_pos = match.end()

        # Add remaining template text
        if current_pos < len(template):
            template_text = template[current_pos:]
            segments.append(PromptSegment(
                segment_type='template_text',
                text=template_text,
                char_start=char_offset + current_pos,
                char_end=char_offset + len(template)
            ))

        return segments

    def _parse_statement_with_delimiters(self, filled_statements: List[str],
                                        full_statement: str,
                                        binding_matrix: BindingMatrix) -> List[PromptSegment]:
        """Parse the full statement including delimiters.

        This matches the actual statement_conjunction_function output
        to identify where delimiters were inserted.
        """
        segments = []
        char_pos = 0

        # Parse each filled statement and track delimiters
        for g, filled in enumerate(filled_statements):
            # Find where this statement appears in the full text
            # Account for capitalization of first word
            if g == 0:
                # First statement is capitalized
                search_text = filled[0].upper() + filled[1:] if filled else ""
            else:
                search_text = filled

            idx = full_statement.find(search_text, char_pos)
            if idx == -1:
                # Try finding without capitalization
                idx = full_statement.find(filled, char_pos)

            if idx >= char_pos:
                # Add delimiter segment if there's text before
                if idx > char_pos:
                    delimiter_text = full_statement[char_pos:idx]
                    segments.append(PromptSegment(
                        segment_type='delimiter',
                        text=delimiter_text,
                        char_start=char_pos,
                        char_end=idx,
                        delimiter_idx=g - 1 if g > 0 else None
                    ))

                # Parse this statement's entities using the FILLED text position
                entity_dict = {
                    f"entity_e{i}": binding_matrix.get_entity(g, i)
                    for i in range(binding_matrix.get_entities_per_group())
                }

                # Create the filled version to parse
                filled_version = self.config.statement_template.format(**entity_dict)

                stmt_segments = self._parse_filled_statement_at_position(
                    filled_version,
                    entity_dict,
                    group_idx=g,
                    char_offset=idx
                )
                segments.extend(stmt_segments)

                char_pos = idx + len(search_text)

        # Add final delimiter (period)
        if char_pos < len(full_statement):
            delimiter_text = full_statement[char_pos:]
            segments.append(PromptSegment(
                segment_type='delimiter',
                text=delimiter_text,
                char_start=char_pos,
                char_end=len(full_statement),
                delimiter_idx=len(filled_statements) - 1
            ))

        return segments

    def _parse_filled_statement_at_position(self, filled_text: str, entity_dict: Dict[str, str],
                                            group_idx: int, char_offset: int) -> List[PromptSegment]:
        """Parse a filled statement text to identify entity positions.

        This works with the FILLED text, not the template, to get accurate positions.
        """
        segments = []
        current_pos = 0

        # For each entity in the entity_dict, find where it appears in the filled text
        # We need to track which entities we've already found to handle duplicates
        entity_positions = []

        for key, entity_value in entity_dict.items():
            if entity_value is None:
                continue

            # Extract entity index from key (e.g., "entity_e0" -> 0)
            if key.startswith("entity_e"):
                entity_idx = int(key.split("_e")[1])
            else:
                continue

            # Find this entity in the filled text (starting from current_pos to maintain order)
            idx = filled_text.find(entity_value, current_pos)
            if idx != -1:
                entity_positions.append((idx, idx + len(entity_value), entity_idx, entity_value))

        # Sort by position
        entity_positions.sort(key=lambda x: x[0])

        # Now create segments for template text and entities
        current_pos = 0
        for start, end, entity_idx, entity_value in entity_positions:
            # Add template text before this entity
            if start > current_pos:
                template_text = filled_text[current_pos:start]
                segments.append(PromptSegment(
                    segment_type='template_text',
                    text=template_text,
                    char_start=char_offset + current_pos,
                    char_end=char_offset + start
                ))

            # Add entity segment
            segments.append(PromptSegment(
                segment_type='entity',
                text=entity_value,
                char_start=char_offset + start,
                char_end=char_offset + end,
                group_idx=group_idx,
                entity_idx=entity_idx
            ))

            current_pos = end

        # Add remaining template text
        if current_pos < len(filled_text):
            template_text = filled_text[current_pos:]
            segments.append(PromptSegment(
                segment_type='template_text',
                text=template_text,
                char_start=char_offset + current_pos,
                char_end=char_offset + len(filled_text)
            ))

        return segments

    def _parse_question(self, binding_matrix: BindingMatrix, query_group: int,
                       query_indices: Tuple[int, ...], answer_index: int) -> Tuple[List[PromptSegment], str]:
        """Parse the question portion of the prompt.

        Returns
        -------
        Tuple[List[PromptSegment], str]
            (segments, question_text)
        """
        # Get question template
        question_template = self.template_processor.select_question_template(
            query_indices, answer_index
        )

        # Fill question template
        question = self.template_processor.fill_question_template(
            question_template, query_group, query_indices, binding_matrix
        )

        # Build entity dict for question (may reference entities from query_group)
        entity_dict = {}
        for e in range(binding_matrix.get_entities_per_group()):
            entity = binding_matrix.get_entity(query_group, e)
            role_name = self.config.entity_roles.get(e, f"entity{e}")
            if entity is not None:
                entity_dict[role_name] = entity

        # Also support {query_entity} placeholder
        if query_indices:
            entity_dict['query_entity'] = binding_matrix.get_entity(
                query_group, query_indices[0]
            )

        # Parse question with entity tracking
        segments = self._parse_question_template(
            question_template, question, entity_dict, query_group, query_indices
        )

        return segments, question

    def _parse_question_template(self, template: str, filled_question: str,
                                 entity_dict: Dict[str, str], query_group: int,
                                 query_indices: Tuple[int, ...]) -> List[PromptSegment]:
        """Parse question template to identify entity positions."""
        segments = []

        # Find placeholders in template
        pattern = r'\{(\w+)\}'

        # We need to match template to filled question
        # Start after statement + space
        # This will be adjusted by caller
        char_offset = 0  # Relative to question start

        current_pos = 0
        filled_pos = 0

        for match in re.finditer(pattern, template):
            placeholder = match.group(1)

            # Add question text before placeholder
            if match.start() > current_pos:
                template_text = template[current_pos:match.start()]
                segments.append(PromptSegment(
                    segment_type='question_text',
                    text=template_text,
                    char_start=char_offset + filled_pos,
                    char_end=char_offset + filled_pos + len(template_text)
                ))
                filled_pos += len(template_text)

            # Identify which entity this is
            entity_value = entity_dict.get(placeholder)
            if entity_value:
                # Determine entity_idx from placeholder
                entity_idx = None
                if placeholder == 'query_entity':
                    entity_idx = query_indices[0] if query_indices else None
                else:
                    # Check entity roles
                    for e_idx, role in self.config.entity_roles.items():
                        if role == placeholder:
                            entity_idx = e_idx
                            break

                segments.append(PromptSegment(
                    segment_type='entity',
                    text=entity_value,
                    char_start=char_offset + filled_pos,
                    char_end=char_offset + filled_pos + len(entity_value),
                    group_idx=query_group,
                    entity_idx=entity_idx
                ))
                filled_pos += len(entity_value)

            current_pos = match.end()

        # Add remaining question text
        if current_pos < len(template):
            remaining = template[current_pos:].format(**entity_dict)
            # Remove any unfilled placeholders
            for key in entity_dict:
                remaining = remaining.replace(f"{{{key}}}", "")

            if remaining:
                segments.append(PromptSegment(
                    segment_type='question_text',
                    text=remaining,
                    char_start=char_offset + filled_pos,
                    char_end=char_offset + filled_pos + len(remaining)
                ))

        return segments


class PromptTokenizer:
    """Compute token positions for parsed prompt segments."""

    def __init__(self, pipeline: LMPipeline):
        self.pipeline = pipeline

    def tokenize_prompt(self, parsed_prompt: ParsedPrompt) -> ParsedPrompt:
        """Add token position information to all segments.

        Parameters
        ----------
        parsed_prompt : ParsedPrompt
            Parsed prompt structure (will be modified in place)

        Returns
        -------
        ParsedPrompt
            The same ParsedPrompt with token_positions filled in
        """
        # Tokenize the full text
        text = parsed_prompt.raw_text
        trace = CausalTrace(
            mechanisms={
                "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
            },
            inputs={"raw_input": text},
        )
        token_ids = self.pipeline.load([trace], add_special_tokens=False)["input_ids"][0]
        token_ids_list = token_ids.tolist()

        # Build character-to-token mapping
        char_to_token = self._build_char_to_token_map(text, token_ids_list)

        # Map each segment's character range to tokens
        for segment in parsed_prompt.segments:
            token_positions = set()
            for char_pos in range(segment.char_start, segment.char_end):
                if char_pos in char_to_token:
                    token_positions.add(char_to_token[char_pos])

            segment.token_positions = sorted(list(token_positions))

        return parsed_prompt

    def _build_char_to_token_map(self, text: str, token_ids: List[int]) -> Dict[int, int]:
        """Build mapping from character positions to token indices.

        Uses the same approach as get_substring_token_ids.
        """
        char_to_token = {}
        reconstructed = ""

        for token_idx, token_id in enumerate(token_ids):
            # Decode up to and including this token
            decoded_so_far = self.pipeline.tokenizer.decode(
                token_ids[:token_idx + 1],
                skip_special_tokens=True
            )

            # Mark new characters as belonging to this token
            start_pos = len(reconstructed)
            end_pos = len(decoded_so_far)

            for char_pos in range(start_pos, end_pos):
                char_to_token[char_pos] = token_idx

            reconstructed = decoded_so_far

        return char_to_token


# High-level API functions

def get_entity_token_indices_structured(
    input_sample: dict,
    pipeline: LMPipeline,
    config: EntityBindingTaskConfig,
    group_idx: int,
    entity_idx: int,
    region: Optional[str] = None,
    token_idx: Optional[int] = None
) -> List[int]:
    """Get token positions for an entity using structured parsing.

    Parameters
    ----------
    input_sample : dict
        Input sample with entity values and raw_input
    pipeline : LMPipeline
        Pipeline with tokenizer
    config : EntityBindingTaskConfig
        Task configuration
    group_idx : int
        Entity group index
    entity_idx : int
        Entity index within group
    region : str, optional
        Which region: 'statement', 'question', or None for first occurrence
    token_idx : int, optional
        If specified, return only the token at this index within the entity

    Returns
    -------
    List[int]
        Token position indices
    """
    # Parse prompt
    parser = PromptParser(config)
    parsed = parser.parse_prompt(input_sample)

    # Tokenize
    tokenizer = PromptTokenizer(pipeline)
    parsed = tokenizer.tokenize_prompt(parsed)

    # Get entity tokens
    token_positions = parsed.get_entity_tokens(group_idx, entity_idx, region)

    # Handle token_idx selection
    if token_idx is not None:
        if token_idx < 0 or token_idx >= len(token_positions):
            raise ValueError(
                f"token_idx {token_idx} out of bounds for entity with {len(token_positions)} tokens"
            )
        return [token_positions[token_idx]]

    return token_positions
