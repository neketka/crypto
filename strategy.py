from enum import Enum
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import scipy.integrate as integ


class StrategyMotion(Enum):
  OPEN = 0,
  CLOSE = 1

class Strategy(ABC):
  @abstractmethod
  def tick(self, open: bool, times: npt.ArrayLike, prices: npt.ArrayLike) -> Optional[StrategyMotion]:
    pass


class StrategyExecutor:
  def __init__(self) -> None:
    self._usd = 0
    self._token = 0
    self._strategies: list[Strategy] = []
    self._transactions: list[tuple[float, float, bool]] = []
    self._x_smooth: npt.ArrayLike = []
    self._y_smooth: npt.ArrayLike = []

  def _smoothData(self, closings: npt.ArrayLike) -> None:
    count = len(closings)
    diff_1 = np.diff(closings)
    x_diff_1 = np.indices(diff_1.shape)[0]
    x_diff_1_int = np.linspace(0, diff_1.shape[0], count * 4)

    diff_1_int = np.interp(x_diff_1_int, x_diff_1, diff_1)
    closings_int = integ.cumtrapz(diff_1_int, x_diff_1_int) + closings[0]

    self._x_smooth = x_diff_1_int[1:] + 0.5
    self._y_smooth = closings_int

  def getTransactions(self) -> list[tuple[float, float, bool]]:
    return self._transactions

  def addStrategy(self, strategy: Strategy) -> None:
    self._strategies.append(strategy)

  def getStrategies(self) -> list[tuple[float, float, bool]]:
    return self._strategies
  
  def getSmoothX(self) -> npt.ArrayLike:
    return self._x_smooth
  
  def getSmoothY(self) -> npt.ArrayLike:
    return self._y_smooth

  def backtest(self, fee: float, closings: npt.ArrayLike) -> tuple[float, float]:
    self._usd = 100
    self._token = 0
    self._smoothData(closings)

    baseline = 100 / self._y_smooth[0] * self._y_smooth[-1] * (1 - fee) ** 2

    for index in range(100, len(self._x_smooth)):
      x = self._x_smooth[index - 100:index]
      y = self._y_smooth[index - 100:index]

      for strategy in self._strategies:
        motion = strategy.tick(self._token > 0, x, y)
        if motion == StrategyMotion.OPEN and self._usd > 0:
          self._token = self._usd / y[-1] * (1 - fee)
          self._usd = 0
          self._transactions.append((self._x_smooth[index], self._y_smooth[index], True))
        elif motion == StrategyMotion.CLOSE and self._token > 0:
          self._usd = self._token * y[-1] * (1 - fee)
          self._token = 0
          self._transactions.append((self._x_smooth[index], self._y_smooth[index], False))

    self._usd += self._token * y[-1] * (1 - fee)
    self._token = 0
    return baseline, self._usd


class LinRegStrategyParams:
  sellTrendThreshold = 2.0
  buyTrendThreshold = 2.0
  shortTermWindow = 10
  longTermWindow = 80
  tradeFee = 0.002
  trailingMargin = 0.02
  lossMargin = 0.01


class LinRegStrategy(Strategy):
  def __init__(self, params: LinRegStrategyParams) -> None:
    self._params = params
    self._sellBar = 0

  def _fitLine(self, x, y):
    coeff = np.polynomial.polynomial.polyfit(x, y, 1)
    return coeff[1], coeff[0]

  def tick(self, open: bool, times: npt.ArrayLike, prices: npt.ArrayLike) -> Optional[StrategyMotion]:
    p = self._params
    
    xShort = times[:-p.shortTermWindow]
    yShort = prices[:-p.shortTermWindow]
    mLinShort, bLinShort = self._fitLine(xShort, yShort)

    xLong = times[:-p.longTermWindow]
    yLong = prices[:-p.longTermWindow]
    mLinLong, bLinLong = self._fitLine(xLong, yLong)

    yFit = mLinLong * times[-1] + bLinLong
    stdFit = np.std(yLong)
    
    price = prices[-1]

    if open and price <= self._sellBar:
      return StrategyMotion.CLOSE
    
    self._sellBar = max(self._sellBar, (1.0 - 2.0 * p.tradeFee - p.trailingMargin) * price)

    sellSignal = mLinLong - mLinShort > p.sellTrendThreshold and yFit - stdFit > price
    buySignal = mLinShort - mLinLong > p.buyTrendThreshold and yFit + stdFit < price

    if sellSignal and open:
      return StrategyMotion.CLOSE
    elif buySignal and not open:
      self._sellBar = price * (1.0 - p.lossMargin + 2.0 * p.tradeFee)
      return StrategyMotion.OPEN