# 青鸟 Python SDK 参考

## 安装与初始化

通过 pip install qingniao-sdk 安装，支持 Python 3.9 及以上版本。初始化客户端：QingniaoClient(api_key, endpoint=None)。api_key 从控制台获取；endpoint 默认指向公有云接入点，私有化部署时改为自己的网关地址。客户端内部维护连接池，进程内应当复用同一个实例，不要在每次调用时新建。

## publish：发布消息

签名：publish(topic, payload, qos=1, ttl=None)。topic 为主题名；payload 支持 bytes 或 str，最大长度受套餐限制；调用成功返回全局唯一的 message_id，失败时抛出 QingniaoError。qos 与 ttl 两个参数的具体取值语义见下一节。

## qos 与 ttl 参数详解

qos=0 表示至多一次：不重试、不确认，适合可容忍丢失的遥测数据。qos=1 表示至少一次：SDK 自动重试直到收到服务端确认，可能重复，是默认值。qos=2 表示精确一次：在 broker 端按 producer_id 与序列号去重，仅 2.3 及以上版本支持，且免费版不可用（免费版仅支持 qos 0 和 1）。ttl 以秒为单位，消息超过 ttl 仍未投递成功则被丢弃并记录到死信队列；ttl=None 表示永不过期。

## subscribe：订阅与回调

签名：subscribe(topic, callback, auto_ack=True)。callback 接收 Message 对象，包含 payload、message_id 与投递次数。auto_ack=True 时回调正常返回即自动确认；设为 False 则需要显式调用 msg.ack()，用于需要先落库再确认的场景。回调内抛出异常会触发重新投递。
