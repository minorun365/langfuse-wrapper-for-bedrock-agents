import os
import json
import uuid
import boto3
from dotenv import load_dotenv
from langfuse import Langfuse
from typing import Optional, Dict, Any
from datetime import datetime

class BedrockLangfuseTracer:
    def __init__(self, debug=False):
        load_dotenv()
        self.bedrock = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        self.langfuse = Langfuse()
        self.debug = debug
        
    def _debug_print(self, message: str, data: Any = None) -> None:
        """デバッグ情報を出力"""
        if self.debug:
            print(f"DEBUG: {message}")
            if data is not None:
                if isinstance(data, (dict, list)):
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                else:
                    print(data)

    def create_base_trace(self, session_id: str, input_text: str) -> Any:
        """基本となるトレースを作成"""
        return self.langfuse.trace(
            id=session_id,
            name="Bedrock Agent Invocation",
            input=input_text,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "agent_id": os.getenv("AGENT_ID"),
                "agent_alias_id": os.getenv("AGENT_ALIAS_ID")
            }
        )

    def _safe_json_loads(self, data: Any) -> Dict:
        """JSONデータを安全にパース"""
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return {"content": data}
        elif isinstance(data, bytes):
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {"content": str(data)}
        elif isinstance(data, dict):
            return data
        return {"content": str(data)}

    def _process_trace_chunk(self, chunk: Dict, parent_span: Any) -> None:
        """トレースチャンクを処理"""
        trace_data = chunk.get("trace", {})
        if "orchestrationTrace" in trace_data.get("trace", {}):
            self._process_orchestration_trace(parent_span, trace_data["trace"]["orchestrationTrace"])

    def _process_orchestration_trace(self, parent_span: Any, orchestration_trace: Dict) -> None:
        """オーケストレーショントレースを処理"""
        # Model Invocation Input
        if "modelInvocationInput" in orchestration_trace:
            input_data = orchestration_trace["modelInvocationInput"]
            generation = parent_span.generation(
                name="model_invocation",
                model="bedrock-agent",
                model_parameters=input_data.get("inferenceConfiguration", {}),
                input=input_data.get("text", ""),
                metadata={
                    "type": input_data.get("type", ""),
                    "trace_id": input_data.get("traceId", "")
                }
            )
            
            # Model Invocation Output
            if "modelInvocationOutput" in orchestration_trace:
                output = orchestration_trace["modelInvocationOutput"]
                raw_response = self._safe_json_loads(output.get("rawResponse", {}).get("content", ""))
                
                # トークン使用量の記録
                usage = {}
                if "metadata" in output and "usage" in output["metadata"]:
                    usage = {
                        "input": output["metadata"]["usage"].get("inputTokens", 0),
                        "output": output["metadata"]["usage"].get("outputTokens", 0),
                        "unit": "TOKENS"
                    }
                
                generation.end(
                    output=json.dumps(raw_response, ensure_ascii=False),
                    usage_details=usage
                )

        # Rationale
        if "rationale" in orchestration_trace:
            parent_span.event(
                name="rationale",
                input=orchestration_trace["rationale"].get("text", ""),
                metadata={
                    "trace_id": orchestration_trace["rationale"].get("traceId", "")
                }
            )

        # Observation
        if "observation" in orchestration_trace:
            self._process_observation(parent_span, orchestration_trace["observation"])

    def _process_observation(self, parent_span: Any, observation: Dict) -> None:
        """オブザベーションを処理"""
        trace_id = observation.get("traceId", "")
        obs_type = observation.get("type", "")

        if "finalResponse" in observation:
            parent_span.event(
                name="final_response",
                input=observation["finalResponse"].get("text", ""),
                metadata={"trace_id": trace_id, "type": obs_type}
            )
        
        if "agentCollaboratorInvocationOutput" in observation:
            output = observation["agentCollaboratorInvocationOutput"]
            collaborator_span = parent_span.span(
                name=f"collaborator_{output.get('agentCollaboratorName', 'unknown')}",
                input=json.dumps(output.get("output", {}), ensure_ascii=False),
                metadata={
                    "collaborator_arn": output.get("agentCollaboratorAliasArn", ""),
                    "trace_id": trace_id
                }
            )
            collaborator_span.end()

    def trace_agent_interaction(self, input_text: str) -> Dict:
        """BedrockエージェントとのやりとりをLangfuseでトレース"""
        session_id = str(uuid.uuid4())
        trace = self.create_base_trace(session_id, input_text)
        
        try:
            # メインのオーケストレーションSpanを作成
            orchestration_span = trace.span(
                name="orchestration",
                input=input_text
            )

            self._debug_print("Invoking Bedrock agent...")
            
            response = self.bedrock.invoke_agent(
                agentId=os.getenv("AGENT_ID"),
                agentAliasId=os.getenv("AGENT_ALIAS_ID"),
                sessionId=session_id,
                enableTrace=True,
                inputText=input_text
            )

            self._debug_print("Processing response stream...")
            
            # チャンクを処理
            for event in response['completion']:
                try:
                    self._process_trace_chunk(event, orchestration_span)
                except Exception as chunk_error:
                    self._debug_print(f"Error processing chunk: {str(chunk_error)}")
                    orchestration_span.event(
                        name="chunk_processing_error",
                        input=str(chunk_error)
                    )

            orchestration_span.end()
            return {"status": "success", "trace_id": trace.id}

        except Exception as e:
            error_msg = str(e)
            self._debug_print(f"Error in trace_agent_interaction: {error_msg}")
            
            if 'orchestration_span' in locals():
                orchestration_span.end(error=error_msg)
                
            trace.update(metadata={"error": error_msg})
            return {"status": "error", "message": error_msg}

# 使用例
if __name__ == "__main__":
    tracer = BedrockLangfuseTracer(debug=True)
    result = tracer.trace_agent_interaction(
        "bedrockエージェントでlangfuseにトレースを送る方法は？"
    )
    print("Result:", result)
