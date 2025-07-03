from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from core.utils import (
    schema_parser,
)
from core.events import (
    SchemaEnrichmentEvent,
)
from core.services import llm_chat_with_pydantic
from response.log_manager import (
    log_step_start,
    log_step_end,
    log_success,
    log_error,
    log_warning,
    log_prompt,
)
from core.templates import QUESTION_SUGGESTION_SKELETON
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel, Field
from typing import List
import logging
import json

logger = logging.getLogger(__name__)

class QuestionSuggestions(BaseModel):
    """Model for suggested questions based on schema."""
    questions: List[str] = Field(..., description="List of insightful questions that can be answered using the database")

class QuestionSuggestionWorkflow(Workflow):
    """Workflow for suggesting insightful questions based on database schema."""

    def __init__(
        self,
        llm: Ollama | GoogleGenAI,
        *args, **kwargs
    ) -> None:
        """Initialize the Question Suggestion Workflow."""
        super().__init__(*args, **kwargs)
        self.llm = llm
        self._timeout = 300.0
        
        # Question suggestion prompt template
        self.question_suggestion_template = QUESTION_SUGGESTION_SKELETON

    @step
    async def Start_workflow(self, context: Context, ev: StartEvent) -> SchemaEnrichmentEvent | StopEvent:
        """Start the Question Suggestion workflow."""
        start_time = log_step_start("START", message="Starting Question Suggestion Workflow")
        
        # Store schema and params in context
        await context.set("table_details", ev.table_details)
        await context.set("top_k", ev.top_k if hasattr(ev, "top_k") else 5)
        
        # Process database schema
        table_details = ev.table_details
        log_step_start("START", total_tables=len(table_details))
        
        # Generate database description
        database_description = ""
        if hasattr(ev, "database_description") and ev.database_description:
            database_description = ev.database_description
            log_step_start("START", message="Using provided database description")
        else:
            log_step_start("START", message="No database description provided. Using schema only.")
        
        log_step_end("START", start_time)
        return SchemaEnrichmentEvent(database_description=database_description, clusters=[table_details])

    @step
    async def Generate_questions(self, context: Context, ev: SchemaEnrichmentEvent) -> StopEvent:
        """Generate insightful questions based on the schema."""
        start_time = log_step_start("GENERATE", message="Generating question suggestions")
        
        # Get parameters from context
        table_details = (await context.get("table_details"))
        top_k = await context.get("top_k")
        
        # Parse schema for prompt
        schema = schema_parser(table_details, "Synthesis", include_sample_data=False)
        log_step_start("GENERATE", message=f"Parsed schema with {len(table_details)} tables")
        
        # Format prompt
        question_prompt = self.question_suggestion_template.format(
            database_description=ev.database_description,
            schema=schema,
            top_k=top_k
        )

        log_prompt(question_prompt, "GENERATE")
        
        # Get suggestions from LLM
        try:
            log_step_start("GENERATE", message=f"Requesting LLM to generate {top_k} questions")
            suggestions = llm_chat_with_pydantic(
                llm=self.llm, 
                prompt=PromptTemplate(question_prompt), 
                pydantic_model=QuestionSuggestions
            )
            
            if not suggestions or not suggestions.questions:
                log_error("GENERATE", "Failed to generate questions")
                return StopEvent(result={"error": "Failed to generate questions"})
            
            # Format response
            response = []
            for i, question in enumerate(suggestions.questions):
                response.append({
                    "id": i+1,
                    "question": question,
                })
                
            log_success("GENERATE", f"Generated {len(response)} question suggestions")
            log_step_end("GENERATE", start_time)
            return StopEvent(result={"suggestions": response})
            
        except Exception as e:
            error_msg = f"Error generating questions: {str(e)}"
            log_error("GENERATE", error_msg)
            log_step_end("GENERATE", start_time)
            return StopEvent(result={"error": error_msg})
