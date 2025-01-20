import os
import json
import uuid
import boto3
from dotenv import load_dotenv
from langfuse import Langfuse
from typing import Optional, Dict, Any
from datetime import datetime

class BedrockLangfuseTracer:
    """
    Amazon Bedrock AgentのトレースをLangfuseに送信するクラス
    
    このクラスは以下の主要な機能を提供します：
    1. Bedrock Agentの実行トレースを取得
    2. 取得したトレースをLangfuseの形式に変換
    3. Langfuseにトレースデータを送信
    
    使用例：
        tracer = BedrockLangfuseTracer(debug=True)
        result = tracer.trace_agent_interaction("あなたの質問")
    """
    def __init__(self, debug=False):
        """
        クラスの初期化
        
        Args:
            debug (bool): デバッグモードの有効/無効を指定
        """
        load_dotenv()  # .envファイルから環境変数を読み込み

        # Bedrock AgentのAPIクライアントを初期化
        self.bedrock = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

        # Langfuseクライアントを初期化
        self.langfuse = Langfuse()
        self.debug = debug
        
    def _debug_print(self, message: str, data: Any = None) -> None:
        """
        デバッグ情報を出力
        
        Args:
            message (str): デバッグメッセージ
            data (Any): 出力したいデータ（dict, list, その他）
        """
        if self.debug:
            print(f"DEBUG: {message}")
            if data is not None:
                if isinstance(data, (dict, list)):
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                else:
                    print(data)

    def create_base_trace(self, session_id: str, input_text: str) -> Any:
        """
        Langfuseの基本トレースを作成
        
        Args:
            session_id (str): セッションID
            input_text (str): ユーザーの入力テキスト
            
        Returns:
            Trace: Langfuseのトレースオブジェクト
        """
        return self.langfuse.trace(
            id=session_id,  # トレースを一意に識別するID
            name="Bedrock Agent Invocation",  # トレースの名前
            input=input_text,  # ユーザー入力
            metadata={  # トレースに関する追加情報
                "timestamp": datetime.now().isoformat(),
                "agent_id": os.getenv("AGENT_ID"),
                "agent_alias_id": os.getenv("AGENT_ALIAS_ID")
            }
        )

    def _safe_json_loads(self, data: Any) -> Dict:
        """
        JSONデータを安全にパース
        様々な形式（文字列、バイト列、辞書）のデータを適切に処理
        
        Args:
            data (Any): パースするデータ
            
        Returns:
            Dict: パース結果の辞書
        """
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
        """
        Bedrock Agentのトレースチャンクを処理
        
        Args:
            chunk (Dict): トレースチャンク
            parent_span (Any): 親となるLangfuseのspan
        """
        # トレースデータを取得
        trace_data = chunk.get("trace", {})
        
        # オーケストレーショントレースがある場合は処理
        if "orchestrationTrace" in trace_data.get("trace", {}):
            self._process_orchestration_trace(parent_span, trace_data["trace"]["orchestrationTrace"])

    def _process_orchestration_trace(self, parent_span: Any, orchestration_trace: Dict) -> None:
        """
        オーケストレーショントレースを処理
        Bedrock Agentの実行プロセスの中心となる部分を処理
        
        Args:
            parent_span (Any): 親となるLangfuseのspan
            orchestration_trace (Dict): オーケストレーショントレース
        """
        # モデルの入力処理
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
            
            # モデルの出力処理
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

        # モデルの推論根拠を処理
        if "rationale" in orchestration_trace:
            parent_span.event(
                name="rationale",
                input=orchestration_trace["rationale"].get("text", ""),
                metadata={
                    "trace_id": orchestration_trace["rationale"].get("traceId", "")
                }
            )

        # モデルの観測結果を処理
        if "observation" in orchestration_trace:
            self._process_observation(parent_span, orchestration_trace["observation"])

    def _process_observation(self, parent_span: Any, observation: Dict) -> None:
        """
        観測結果を処理
        モデルの最終的な出力や協力エージェントの結果を処理
        
        Args:
            parent_span (Any): 親となるLangfuseのspan
            observation (Dict): 観測結果
        """
        trace_id = observation.get("traceId", "")
        obs_type = observation.get("type", "")

        # 最終的な応答を処理
        if "finalResponse" in observation:
            parent_span.event(
                name="final_response",
                input=observation["finalResponse"].get("text", ""),
                metadata={"trace_id": trace_id, "type": obs_type}
            )
        
        # 協力エージェントからの出力を処理
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
        """
        BedrockエージェントとのやりとりをLangfuseでトレース
        このメソッドが本クラスのメインのエントリーポイント
        
        Args:
            input_text (str): ユーザーからの入力テキスト
            
        Returns:
            Dict: 処理結果（成功時は trace_id を、失敗時はエラーメッセージを含む）
        """
        session_id = str(uuid.uuid4())
        trace = self.create_base_trace(session_id, input_text)
        
        try:
            # メインのオーケストレーションSpanを作成
            orchestration_span = trace.span(
                name="orchestration",
                input=input_text
            )

            self._debug_print("Invoking Bedrock agent...")
            
            # Bedrock Agentを呼び出し
            response = self.bedrock.invoke_agent(
                agentId=os.getenv("AGENT_ID"),
                agentAliasId=os.getenv("AGENT_ALIAS_ID"),
                sessionId=session_id,
                enableTrace=True,
                inputText=input_text
            )

            self._debug_print("Processing response stream...")
            
            # レスポンスストリームの各チャンクを処理
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
            
            # エラーが発生した場合でもspanを適切に終了
            if 'orchestration_span' in locals():
                orchestration_span.end(error=error_msg)
                
            trace.update(metadata={"error": error_msg})
            return {"status": "error", "message": error_msg}

# 使用例
if __name__ == "__main__":

    # デバッグモードを有効にしてトレーサーを初期化
    tracer = BedrockLangfuseTracer(debug=True)

    # Bedrock Agentにクエリを送信し、その実行をトレース
    result = tracer.trace_agent_interaction(
        "bedrockエージェントでlangfuseにトレースを送る方法は？"
    )
    print("Result:", result)
