from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


class Grpc_client(object):
    def __init__(self, host, port, creds=None,
                 channel_type="insecure_channel"):

        if channel_type == "insecure_channel":
            channel = implementations.insecure_channel(host, int(port))
        elif channel_type == "secure_channel":
            channel = implementations.secure_channel(host, int(port), cred)
        else:
            print 'Invalid channel_type'
            exit(1)

        self._stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel)

    def set_request(self,
                    name='default',
                    signature_name='predict_images',
                    inputs=None,
                    version=None,
                    output_filter=None):

        if not isinstance(inputs, dict):
            print 'inputs is not a dict.'
            exit(1)

        if (output_filter is not None) and (not isinstance(
                output_filter, list)):
            print 'output_filter must be None or a str list.'
            exit(1)

        request = predict_pb2.PredictRequest()

        request.model_spec.name = name
        if version is not None:
            request.model_spec.version.value = version
        request.model_spec.signature_name = signature_name

        for key in inputs:
            request.inputs[key].CopyFrom(inputs[key])

        if output_filter != None:
            for filter_o in output_filter:
                if not isinstance(filter_o, str):
                    raise ValueError(
                        "One member of output_filter is not a str: {0}.".
                        format(filter_o))
                predict_req.output_filter.append(filter_o)
        return request

    def make_prediction(self, request, timeout=5.0):
        return self._stub.Predict(request, timeout)

    def make_classification(self, request, timeout=5.0):
        return self._stub.Classify(request, timeout)
