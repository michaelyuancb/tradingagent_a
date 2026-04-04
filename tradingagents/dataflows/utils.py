import os
import json
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Annotated

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    """
    保存输出内容。
    
    参数：
        data: 输入数据。
        tag: 保存输出时使用的短标签。
        save_path: 输出文件的可选保存路径。
    
    返回：
        None: 无返回值。
    """
    if save_path:
        data.to_csv(save_path, encoding="utf-8-sig")
        print(f"{tag} saved to {save_path}")


def get_current_date():
    """
    返回当前日期。
    
    返回：
        Any: 当前查询结果。
    """
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    """
    为类中方法统一添加装饰器。
    
    参数：
        decorator: 需要应用到目标类方法上的装饰器。
    
    返回：
        None: 无返回值。
    """
    def class_decorator(cls):
        """
        执行类装饰器逻辑。
        
        返回：
            None: 无返回值。
        """
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    """
    返回下一个工作日。
    
    参数：
        date: 需要规范化的日期或 datetime 值。
    
    返回：
        Any: 当前查询结果。
    """
    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date
