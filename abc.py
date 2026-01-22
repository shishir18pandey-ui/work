apiVersion: v1
kind: Pod
metadata:
  name: tracepusher-test
  namespace: tracing
spec:
  restartPolicy: Never
  imagePullSecrets:
    - name: artifactory-registry-ha
  containers:
    - name: tracepusher
      image: artifactory.idfcfirstbank.com/infra-common-docker/gardnera/tracepusher:v0.8.0
      args:
        # OTLP Collector endpoint
        - "-ep"
        - "http://ot-collector.tracing.svc.cluster.local:4318"

        # Service & span identity
        - "-sen"
        - "sop-tracepusher-test"
        - "-spn"
        - "pushed-span"
        - "-dur"
        - "5"

        # -------- Span Attributes (mapped) --------
        - "-attr"
        - "http.method=GET"
        - "-attr"
        - "http.route=/login"
        - "-attr"
        - "http.status_code=307"
        - "-attr"
        - "http.scheme=http"
        - "-attr"
        - "http.host=127.0.0.1:8000"
        - "-attr"
        - "http.user_agent=Mozilla/5.0"
        - "-attr"
        - "net.peer.ip=127.0.0.1"

        # -------- Custom / business context --------
        - "-attr"
        - "component=tracepusher"
        - "-attr"
        - "source=synthetic-test"

      env:
        # Resource-level attributes (service metadata)
        - name: OTEL_RESOURCE_ATTRIBUTES
          value: >
            service.namespace=ai-platform,
            deployment.environment=uat,
            k8s.namespace.name=tracing,
            k8s.pod.name=tracepusher-test








see right now im not getting any trace or trace connection neither im able to see in jager ui , i asked my senior so the  he told me like by creating this file , he created one test service and also got the tcae exported,
this is what she told :
kubectl apply -f tracepusherpod.yml -n tracing
tracepusherpod.yml
 
thanks 
 
 
Either use SimpleSpanProcessor (flush immediately) - u can also try this in ur code - to flush trace immediately instead of batch - we can see if this works
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
tracer_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))

