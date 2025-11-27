"""
부도 예측 모델 로딩 및 예측

Part3 노트북과 동일한 파이프라인으로 예측 수행
학습된 모델을 로드하고 새로운 데이터에 대해 예측 수행
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import sys

# 전처리 모듈 import (pickle 로딩을 위해 클래스들 import 필요)
try:
    from src.preprocessing.transformers import (
        create_preprocessing_pipeline,
        InfiniteHandler,
        LogTransformer,
        Winsorizer
    )
except ImportError:
    # deployment 폴더에서 실행될 경우
    try:
        from preprocessing.transformers import (
            create_preprocessing_pipeline,
            InfiniteHandler,
            LogTransformer,
            Winsorizer
        )
    except ImportError:
        create_preprocessing_pipeline = None
        InfiniteHandler = None
        LogTransformer = None
        Winsorizer = None
        logging.warning("전처리 모듈을 import할 수 없습니다. 기본 전처리 사용")

logger = logging.getLogger(__name__)


class BankruptcyPredictor:
    """
    부도 예측 모델

    Part3 노트북과 동일한 파이프라인 지원:
    - 전처리 파이프라인 (InfiniteHandler, LogTransformer, Scaler 등)
    - 전체 파이프라인 (전처리 + 모델)
    - 휴리스틱 방식 (모델 없을 때)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        pipeline_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        use_pipeline: bool = True
    ):
        """
        Args:
            model_path: 모델 파일 경로 (단독 모델)
            pipeline_path: 파이프라인 파일 경로 (전처리 + 모델)
            scaler_path: 스케일러 파일 경로 (단독 스케일러)
            use_pipeline: 파이프라인 사용 여부 (Part3 방식)
        """
        self.model = None
        self.pipeline = None
        self.scaler = None
        self.preprocessing_pipeline = None

        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.scaler_path = scaler_path
        self.use_pipeline = use_pipeline
        self.expected_features = None

    def load_model(self):
        """
        모델 로드 (우선순위):
        1. 전체 파이프라인 (전처리 + 모델) - Part3 방식
        2. 모델 + 스케일러 분리
        3. 휴리스틱 방식
        """
        try:
            # 1. 전체 파이프라인 로드 시도 (Part3 방식)
            if self.use_pipeline and self.pipeline_path and self.pipeline_path.exists():
                logger.info(f"📦 전체 파이프라인 로딩 중: {self.pipeline_path}")
                self.pipeline = joblib.load(self.pipeline_path)
                logger.info("✓ Part3 파이프라인 로드 성공!")
                
                if hasattr(self.pipeline, 'steps'):
                    logger.info(f"   파이프라인 단계: {len(self.pipeline.steps)}개")
                    for step_name, _ in self.pipeline.steps:
                        logger.info(f"   - {step_name}")
                elif hasattr(self.pipeline, 'estimators_'):
                    logger.info(f"   모델 타입: VotingClassifier (estimators: {len(self.pipeline.estimators_)})")
                else:
                    logger.info(f"   모델 타입: {type(self.pipeline).__name__}")
                return

            # 2. 모델 단독 로드
            if self.model_path and self.model_path.exists():
                logger.info(f"🎯 모델 로딩 중: {self.model_path}")
                self.model = joblib.load(self.model_path)
                logger.info("✓ 모델 로드 성공")
            else:
                logger.warning("모델 파일을 찾을 수 없습니다.")
                self.model = None

            # 3. 스케일러 로드
            if self.scaler_path and self.scaler_path.exists():
                logger.info(f"📏 스케일러 로딩 중: {self.scaler_path}")
                self.scaler = joblib.load(self.scaler_path)
                logger.info("✓ 스케일러 로드 성공")
            else:
                logger.warning("스케일러 파일을 찾을 수 없습니다.")

                # 스케일러 없으면 전처리 파이프라인 생성
                if create_preprocessing_pipeline:
                    logger.info("기본 전처리 파이프라인 생성 중...")
                    self.preprocessing_pipeline = create_preprocessing_pipeline(
                        use_log_transform=True,
                        use_winsorizer=False,
                        scaler_type='robust'
                    )
                    logger.info("✓ Part3 전처리 파이프라인 생성 완료")

        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            logger.warning("휴리스틱 방식으로 전환합니다.")
            self.model = None
            self.pipeline = None
            self.scaler = None

    def predict(self, features_df: pd.DataFrame) -> Dict:
        """
        부도 확률 예측

        Args:
            features_df: 특성 DataFrame (1행)

        Returns:
            {
                'bankruptcy_probability': 0.15,
                'risk_level': '주의',
                'confidence': 0.85,
                'features_used': [...],
                'model_info': {...}
            }
        """
        try:
            # 1. 전체 파이프라인 사용 (Part3 방식)
            if self.pipeline is not None:
                logger.info("Part3 파이프라인으로 예측 중...")
                X = self._prepare_features(features_df)

                # 파이프라인으로 직접 예측
                if hasattr(self.pipeline, 'predict_proba'):
                    proba = self.pipeline.predict_proba(X)[0]
                    bankruptcy_prob = proba[1]
                    confidence = max(proba)
                else:
                    prediction = self.pipeline.predict(X)[0]
                    bankruptcy_prob = 0.8 if prediction == 1 else 0.2
                    confidence = 0.7

                # 파이프라인 내부 모델 추출 (SHAP용)
                # Part4 노트북 방식: Pipeline의 마지막 단계 (CatBoost) 추출
                if hasattr(self.pipeline, 'steps'):
                    # Pipeline의 마지막 단계가 classifier (CatBoost)
                    model_for_shap = self.pipeline.steps[-1][1]
                    logger.info(f"   - Pipeline에서 최종 모델 추출: {type(model_for_shap).__name__}")

                    # VotingClassifier인 경우 SHAP 계산 스킵
                    if hasattr(model_for_shap, 'estimators_'):
                        logger.warning("VotingClassifier는 SHAP TreeExplainer 미지원 - SHAP 계산 생략")
                        model_for_shap = None

                elif hasattr(self.pipeline, 'named_steps'):
                    model_for_shap = self.pipeline.named_steps.get('classifier', self.pipeline)
                    if hasattr(model_for_shap, 'estimators_'):
                        logger.warning("VotingClassifier는 SHAP TreeExplainer 미지원 - SHAP 계산 생략")
                        model_for_shap = None
                else:
                    model_for_shap = self.pipeline
                    if hasattr(model_for_shap, 'estimators_'):
                        logger.warning("VotingClassifier는 SHAP TreeExplainer 미지원 - SHAP 계산 생략")
                        model_for_shap = None

                # SHAP 계산용 데이터는 전처리된 데이터 (Pipeline 입력과 동일)
                X_for_shap = X

            # 2. 전처리 파이프라인 + 모델 분리 사용
            elif self.preprocessing_pipeline is not None and self.model is not None:
                logger.info("전처리 파이프라인 + 모델로 예측 중...")
                X = self._prepare_features(features_df)
                X_preprocessed = self.preprocessing_pipeline.transform(X)

                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X_preprocessed)[0]
                    bankruptcy_prob = proba[1]
                    confidence = max(proba)
                else:
                    prediction = self.model.predict(X_preprocessed)[0]
                    bankruptcy_prob = 0.8 if prediction == 1 else 0.2
                    confidence = 0.7

                model_for_shap = self.model
                X_for_shap = X_preprocessed

            # 3. 모델만 사용 (스케일러 포함)
            elif self.model is not None:
                logger.info("모델 단독 예측 중...")
                X = self._prepare_features(features_df)

                # 스케일링
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = X

                # 예측
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X_scaled)[0]
                    bankruptcy_prob = proba[1]
                    confidence = max(proba)
                else:
                    prediction = self.model.predict(X_scaled)[0]
                    bankruptcy_prob = 0.8 if prediction == 1 else 0.2
                    confidence = 0.7

                model_for_shap = self.model
                X_for_shap = X_scaled

            # 4. 모델이 없으면 휴리스틱
            else:
                logger.warning("모델 없음. 휴리스틱 기반 예측 사용")
                return self._heuristic_prediction(features_df)

            # SHAP 값 계산
            shap_values = None
            shap_base_value = None
            try:
                import shap
                # CatBoost는 리스트 반환 → 부도(1) 클래스만 사용
                logger.info(f"X_for_shap shape: {X_for_shap.shape}")
                logger.info(f"X_for_shap dtypes: {X_for_shap.dtypes}")
                
                if model_for_shap is not None:
                    logger.info(f"Creating TreeExplainer for {type(model_for_shap)}")
                    try:
                        explainer = shap.TreeExplainer(model_for_shap)
                        logger.info("Calculating shap_values...")
                        shap_values_result = explainer.shap_values(X_for_shap)
                        logger.info("shap_values calculated.")
                    except Exception as e:
                        logger.warning(f"TreeExplainer 초기화 실패: {e}. SHAP 계산을 건너뜁니다.")
                        raise ValueError(f"SHAP 초기화 실패: {e}")
                else:
                    logger.info("SHAP 계산 생략 (VotingClassifier는 미지원)")
                    raise ValueError("VotingClassifier는 SHAP TreeExplainer 미지원")

                logger.info(f"SHAP result type: {type(shap_values_result)}")
                logger.info(f"Expected value type: {type(explainer.expected_value)}")
                logger.info(f"Expected value: {explainer.expected_value}")

                # Part4 노트북 방식: CatBoost는 리스트 반환 → [클래스0, 클래스1]
                if isinstance(shap_values_result, list):
                    # CatBoost: shap_values_result = [array(...), array(...)]
                    # shap_values_result[1] = 부도(클래스 1)에 대한 SHAP 값
                    # shap_values_result[1][0] = 첫 번째 샘플 (shape: (27,))
                    try:
                        shap_values = shap_values_result[1][0]  # numpy 배열 (27개 특성)
                        logger.info(f"CatBoost SHAP values (클래스 1): shape {shap_values.shape}")
                    except IndexError:
                        shap_values = shap_values_result[0][0]
                        logger.warning("클래스 1 없음, 클래스 0 사용")

                    # expected_value도 리스트: [클래스0 기준값, 클래스1 기준값]
                    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
                        shap_base_value = float(explainer.expected_value[1])  # 클래스 1 기준값
                    else:
                        shap_base_value = float(explainer.expected_value)

                    logger.info(f"SHAP base value (클래스 1): {shap_base_value:.4f}")

                else:
                    # 단일 배열인 경우 (이진 분류 단일 출력)
                    if len(shap_values_result.shape) > 1:
                         # (samples, features) - 첫 번째 샘플 선택
                         shap_values = shap_values_result[0]
                    else:
                         # (features,) - 그대로 사용
                         shap_values = shap_values_result

                    shap_base_value = float(explainer.expected_value)

                logger.info("✓ SHAP 값 계산 완료")
            except Exception as e:
                logger.warning(f"SHAP 계산 실패: {e}")
                shap_values = None
                shap_base_value = None

            # 결과 생성
            from src.utils.helpers import get_risk_level
            risk_level, icon, msg = get_risk_level(bankruptcy_prob)

            # 모델 타입 결정
            if self.pipeline is not None:
                model_type = f"Pipeline({type(model_for_shap).__name__})"
            elif self.model is not None:
                model_type = type(self.model).__name__
            else:
                model_type = "Heuristic"

            result = {
                'bankruptcy_probability': float(bankruptcy_prob),
                'risk_level': risk_level,
                'risk_icon': icon,
                'risk_message': msg,
                'confidence': float(confidence),
                'features_used': list(X_for_shap.columns) if hasattr(X_for_shap, 'columns') else [],
                'model_info': {
                    'model_type': model_type,
                    'n_features': X_for_shap.shape[1] if hasattr(X_for_shap, 'shape') else 0
                }
            }

            # SHAP 정보 추가
            if shap_values is not None:
                result['shap_values'] = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
                result['shap_base_value'] = float(shap_base_value)
                result['feature_names'] = list(X_for_shap.columns) if hasattr(X_for_shap, 'columns') else []

            logger.info(f"예측 완료: 부도 확률 {bankruptcy_prob:.1%}, 등급 {risk_level}")

            return result

        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            # 에러 시 휴리스틱 예측
            return self._heuristic_prediction(features_df)

    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        모델에 맞게 특성 준비

        Args:
            features_df: 생성된 특성 DataFrame

        Returns:
            모델 입력용 DataFrame
        """
        # 모델이 기대하는 특성 목록 (26개)
        # Part3 노트북에서 '이해관계자_불신지수' 제거됨 (논리적 오류 방지)
        expected_features = [
            '순부채비율', '운전자본', '운전자본비율', '이자부담률',
            '공공정보리스크', '판관비효율성', '재고회전율', '유동성압박지수', '매출총이익률',
            'OCF_대_유동부채', '부채레버리지', '재고보유일수', '현금소진일수', '매출집중도',
            '연체심각도', '신용등급점수', '부채상환년수', '매출채권_이상지표', '매출채권회전율',
            '총발생액', '현금흐름품질', '긴급유동성', '즉각지급능력', '운전자본_대_자산',
            '이자보상배율', '현금창출능력'
        ]

        X = features_df.copy()
        
        # 1. 특성 이름 매핑 (생성된 특성 -> 모델 기대 특성)
        # 도메인 특성 생성 시 이름과 모델 학습 시 사용한 이름의 차이를 보정
        rename_map = {
            'OCF유동부채비율': 'OCF_대_유동부채',
            '긴급유동성비율': '긴급유동성',
            '유동성위기지수': '유동성압박지수',
            '재무레버리지': '부채레버리지',
            '재고자산회전일수': '재고보유일수',
            '현금흐름적정성': '현금흐름품질',
            '당좌비율': '즉각지급능력',
            '단기지급능력': '현금창출능력',
        }
        X = X.rename(columns=rename_map)
        
        # 중복된 컬럼 제거 (매핑으로 인해 중복 발생 시 첫 번째 것 유지)
        X = X.loc[:, ~X.columns.duplicated()]

        # 2. 누락된 특성 채우기 (기본값 사용)
        # DART API에서 얻을 수 없는 신용평가 정보는 안전한 기본값 사용
        # 보수적 가정: 평균적이고 문제없는 기업으로 가정하여 부도 위험을 과소평가하지 않도록 함
        defaults = {
            # 신용평가 정보 (DART API 미제공, 외부 신용평가사 데이터 필요)
            '신용등급점수': 5.0,        # BBB 등급 (중간 등급, 1~10 스케일에서 5)
            '연체심각도': 0.0,          # 연체 없음 가정 (0 = 연체 없음, 1 = 심각)
            '공공정보리스크': 0.0,      # 세금체납 없음 가정 (0 = 없음, 1 = 있음)
        }
        
        for feature in expected_features:
            if feature not in X.columns:
                if feature in defaults:
                    val = defaults[feature]
                    # Series일 경우 값만 추출
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    X[feature] = val
                    logger.warning(f"특성 '{feature}' 누락됨. 기본값 {val} 사용")
                else:
                    # 매핑되지 않은 나머지 특성은 0으로 채움
                    X[feature] = 0.0
                    logger.warning(f"특성 '{feature}' 누락됨. 0.0으로 채움")

        # 3. 순서 맞추기 및 선택
        X = X[expected_features]

        # 범주형 변수 제거 (숫자형만) - 이미 위에서 선택했으므로 불필요할 수 있으나 안전장치
        X = X.select_dtypes(include=[np.number])

        # NaN/Inf 제거
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        return X

    def _heuristic_prediction(self, features_df: pd.DataFrame) -> Dict:
        """
        휴리스틱 기반 부도 확률 예측 (모델 없을 때)

        주요 지표들을 조합하여 경험적으로 부도 확률 추정

        Args:
            features_df: 특성 DataFrame

        Returns:
            예측 결과
        """
        logger.info("휴리스틱 기반 예측 실행")

        # 주요 위험 지표 추출
        유동성위기 = features_df.get('유동성위기지수', pd.Series([0.5])).iloc[0]
        지급불능위험 = features_df.get('지급불능위험지수', pd.Series([0.5])).iloc[0]
        재무조작위험 = features_df.get('재무조작위험지수', pd.Series([0.3])).iloc[0]

        # 조기경보신호
        경보신호수 = features_df.get('조기경보신호수', pd.Series([0])).iloc[0]

        # 종합 부도 위험 스코어 (가중평균)
        bankruptcy_prob = (
            0.35 * 유동성위기 +
            0.35 * 지급불능위험 +
            0.20 * 재무조작위험 +
            0.10 * min(1.0, 경보신호수 / 5)
        )

        # 0~1 범위로 클리핑
        bankruptcy_prob = max(0.0, min(1.0, bankruptcy_prob))

        from src.utils.helpers import get_risk_level
        risk_level, icon, msg = get_risk_level(bankruptcy_prob)

        result = {
            'bankruptcy_probability': float(bankruptcy_prob),
            'risk_level': risk_level,
            'risk_icon': icon,
            'risk_message': msg,
            'confidence': 0.7,  # 휴리스틱이므로 신뢰도 낮음
            'features_used': ['유동성위기지수', '지급불능위험지수', '재무조작위험지수', '조기경보신호수'],
            'model_info': {
                'model_type': 'Heuristic',
                'n_features': 4,
                'note': '학습된 모델이 없어 경험적 규칙 기반으로 예측했습니다.'
            }
        }

        logger.info(f"휴리스틱 예측 완료: 부도 확률 {bankruptcy_prob:.1%}")

        return result
        return result

    def _parse_shap_value(self, value) -> float:
        """
        SHAP 값 파싱 (float, string, list string 등 처리)
        """
        if value is None:
            return 0.0
            
        if isinstance(value, (float, int, np.number)):
            return float(value)
            
        if isinstance(value, (list, np.ndarray)):
            # 리스트나 배열인 경우 첫 번째 요소 재귀 처리
            if len(value) > 0:
                return self._parse_shap_value(value[0])
            return 0.0
            
        if isinstance(value, (str, np.str_)):
            import ast
            try:
                # 1. 단순 float 변환
                return float(value)
            except:
                try:
                    # 2. 리스트 형태 문자열 파싱 ('[0.123]')
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        return self._parse_shap_value(parsed[0])
                    return float(parsed)
                except:
                    try:
                        # 3. 괄호 제거 후 변환
                        clean_val = value.replace('[', '').replace(']', '').strip()
                        return float(clean_val)
                    except:
                        logger.warning(f"SHAP 값 파싱 실패: {value}")
                        return 0.0
        
        return 0.0
