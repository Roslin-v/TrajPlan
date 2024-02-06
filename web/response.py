# 响应码 200 00 1
# 200 成功    400 客户端异常   500 服务端异常
# 00 01 不同的业务
# 0/1   业务执行成功与否
msgs = {200000: '成功加载页面',
        200010: '登录失败', 200011: '登录成功',
        200020: '注册失败', 200021: '注册成功'}


class Response:
    def __init__(self, code, data=None):
        self.code = code
        self.data = data
        self.msg = msgs[code]

    def res2dict(self):
        return {'code': self.code, 'data': self.data, 'msg': self.msg}
