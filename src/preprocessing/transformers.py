"""
Part3 노트북의 커스텀 Transformer들

Streamlit 앱에서 노트북과 동일한 전처리를 수행하기 위한 transformer
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class InfiniteHandler(BaseEstimator, TransformerMixin):
    """
    무한대 값을 0으로 변환

    재무 비율 계산 시 발생하는 inf, -inf를 안전하게 처리
    """

    def fit(self, X, y=None):
        """학습 (파라미터 없음)"""
        return self

    def transform(self, X):
        """무한대 값을 0으로 변환"""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X = X.replace([np.inf, -np.inf], 0)
        else:
            X = np.copy(X)
            X[np.isinf(X)] = 0
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    로그 변환 (양수 값만)

    왜도(skewness)가 높은 특성을 정규분포에 가깝게 변환
    """

    def __init__(self, threshold: float = 0):
        """
        Args:
            threshold: 이 값보다 큰 값만 로그 변환 (기본: 0)
        """
        self.threshold = threshold
        self.columns_to_transform_ = None

    def fit(self, X, y=None):
        """양수 값만 있는 컬럼 파악"""
        if isinstance(X, pd.DataFrame):
            # 모든 값이 threshold보다 큰 컬럼만 변환
            self.columns_to_transform_ = [
                col for col in X.columns
                if (X[col] > self.threshold).all()
            ]
        return self

    def transform(self, X):
        """로그 변환 (log1p 사용)"""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            
            # Backward compatibility for pickled models
            cols = getattr(self, 'columns_to_transform_', None)
            if cols is None:
                cols = getattr(self, 'positive_cols_', None)
                
            if cols:
                for col in cols:
                    if col in X.columns:
                        X[col] = np.log1p(X[col])
        else:
            # numpy array인 경우 모든 양수에 적용
            X = np.copy(X)
            mask = X > self.threshold
            X[mask] = np.log1p(X[mask])
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorizing - 이상치를 특정 백분위수로 클리핑

    극단값의 영향을 줄이면서 데이터 분포 유지
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        """
        Args:
            lower: 하한 백분위수 (기본: 1%)
            upper: 상한 백분위수 (기본: 99%)
        """
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        """각 특성의 백분위수 계산"""
        if isinstance(X, pd.DataFrame):
            self.lower_bounds_ = X.quantile(self.lower)
            self.upper_bounds_ = X.quantile(self.upper)
        else:
            self.lower_bounds_ = np.percentile(X, self.lower * 100, axis=0)
            self.upper_bounds_ = np.percentile(X, self.upper * 100, axis=0)
        return self

    def transform(self, X):
        """이상치를 백분위수 값으로 클리핑"""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X = X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        else:
            X = np.copy(X)
            X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X


def create_preprocessing_pipeline(
    use_log_transform: bool = True,
    use_winsorizer: bool = False,
    scaler_type: str = 'robust'
):
    """
    Part3 노트북과 동일한 전처리 파이프라인 생성

    Args:
        use_log_transform: 로그 변환 사용 여부
        use_winsorizer: Winsorizer 사용 여부
        scaler_type: 스케일러 타입 ('robust', 'standard', None)

    Returns:
        sklearn.pipeline.Pipeline
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, StandardScaler

    steps = [
        ('inf_handler', InfiniteHandler()),
        ('imputer', SimpleImputer(strategy='median')),
    ]

    if use_log_transform:
        steps.append(('log_transform', LogTransformer()))

    if use_winsorizer:
        steps.append(('winsorizer', Winsorizer()))

    if scaler_type == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif scaler_type == 'standard':
        steps.append(('scaler', StandardScaler()))

    return Pipeline(steps)
