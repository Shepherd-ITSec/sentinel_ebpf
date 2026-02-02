{{- define "sentinel-ebpf.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "sentinel-ebpf.fullname" -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "sentinel-ebpf.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "sentinel-ebpf.labels" -}}
helm.sh/chart: {{ include "sentinel-ebpf.chart" . }}
app.kubernetes.io/name: {{ include "sentinel-ebpf.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "sentinel-ebpf.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sentinel-ebpf.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "sentinel-ebpf.serviceAccountName" -}}
{{- if .Values.probe.serviceAccount.create -}}
{{- default (include "sentinel-ebpf.fullname" .) .Values.probe.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.probe.serviceAccount.name -}}
{{- end -}}
{{- end -}}
