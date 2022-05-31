
import logging
import os


class Logger:
    def __init__(self, name, log_path, mode='a'):
        """
        程序运行日志类的构造函数
        :param name: 需要保存的日志文件名称，默认后缀名称为 .log
        :param log_path: 需要保存的日志文件路径
        :param mode: 日志写入模式， a:追加， w:覆盖
        使用说明：
            1、创建日志实例对象
                logger = Logger("textCNN_train", log_path="../logs").get_log
            2、将关键信息通过日志实例对象写入日志文件
                logger.info("")
        """
        self.__name = name
        self.logger = logging.getLogger(self.__name)
        self.logger.setLevel(logging.DEBUG)
        self.log_path = log_path
        self.mode = mode

        # 创建一个handler，用于写入日志文件
        # log_path = os.path.dirname(os.path.abspath(__file__))
        # 指定utf-8格式编码，避免输出的日志文本乱码
        logname = os.path.join(self.log_path, self.__name + '.log')  # 指定输出的日志文件名
        # 定义handler的输出格式
        formatter = logging.Formatter(
            '%(asctime)s-%(filename)s-[日志信息]-[%(module)s-%(funcName)s-line:%(lineno)d]-%(levelname)s: %(message)s')

        fh = logging.FileHandler(logname, mode=self.mode, encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
        fh.setLevel(logging.DEBUG)

        # 创建一个handler，用于将日志输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    @property
    def get_log(self):
        """定义一个函数，回调logger实例"""
        return self.logger
