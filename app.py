"""
Sistema de OCR Híbrido com Streamlit
Versão: 5.0 - Streamlit Cloud Ready

Funcionalidades:
- Upload de PDFs
- Processamento com fallback inteligente
- Visualização de resultados em tempo real
- Download de outputs
- Estatísticas detalhadas
"""

import streamlit as st
import os
import base64
import time
import json
import requests
from io import BytesIO
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
import tempfile

# Configuração da página
st.set_page_config(
    page_title="OCR Híbrido Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== INSTALAÇÃO DE DEPENDÊNCIAS ====================
@st.cache_resource
def instalar_dependencias():
    """Instala dependências pesadas apenas uma vez"""
    import subprocess
    import sys
    
    packages = [
        "pdf2image",
        "PyPDF2",
        "Pillow",
        "numpy",
        "scikit-learn",
        "tqdm"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

with st.spinner("🔧 Verificando dependências..."):
    instalar_dependencias()

from pdf2image import convert_from_bytes
from PIL import Image, ImageStat
import numpy as np

# ==================== CONFIGURAÇÕES ====================
class Config:
    """Configurações centralizadas"""
    DPI = 150
    MAX_FILE_SIZE = 50  # MB
    
    MODELOS = {
        'gemini_25': {
            'nome': 'Gemini 2.5 Pro',
            'id': 'gemini-2.5-pro',
            'tipo': 'gemini',
            'prioridade': 1
        },
        'gemini_20': {
            'nome': 'Gemini 2.5 Flash',
            'id': 'gemini-2.5-flash',
            'tipo': 'gemini',
            'prioridade': 2
        },
        'groq_maverick': {
            'nome': 'Groq Llama 4 Maverick',
            'id': 'meta-llama/llama-4-maverick-17b-128e-instruct',
            'tipo': 'groq',
            'prioridade': 3
        },
        'groq_scout': {
            'nome': 'Groq Llama 4 Scout',
            'id': 'meta-llama/llama-4-scout-17b-16e-instruct',
            'tipo': 'groq',
            'prioridade': 4
        }
    }
    
    ESTRATEGIAS = {
        'muito_complexa': {  # Score ≥ 80
            'min_score': 80,
            'descricao': 'MUITO COMPLEXA',
            'sequencia': [
                {'modelo': 'gemini_25', 'retries': 10},
                {'modelo': 'gemini_20', 'retries': 5},
                {'modelo': 'groq_maverick', 'retries': 5},
            ]
        },
        'media': {  # Score 50-79
            'min_score': 50,
            'descricao': 'MÉDIA',
            'sequencia': [
                {'modelo': 'gemini_25', 'retries': 3},
                {'modelo': 'gemini_20', 'retries': 5},
                {'modelo': 'groq_maverick', 'retries': 5},
            ]
        },
        'simples': {  # Score < 50
            'min_score': 0,
            'descricao': 'SIMPLES',
            'sequencia': [
                {'modelo': 'groq_maverick', 'retries': 5},
                {'modelo': 'groq_scout', 'retries': 5},
            ]
        }
    }

# ==================== PROMPTS ====================
PROMPT_SIMPLES = """Extraia TODO o texto desta página de forma ESTRUTURADA.

INSTRUÇÕES:
✅ Mantenha formatação (numeração, hierarquia)
✅ Inclua títulos, parágrafos, listas, tabelas
✅ NÃO descreva imagens

FORMATO: Markdown"""

PROMPT_COMPLEXO = """Analise esta página:

📝 TEXTO:
Extraia todo texto visível

🖼️ IMAGENS:
Para cada imagem:
- TIPO: [tipo]
- DESCRIÇÃO: [detalhes]
- CONTEXTO: [relação]"""

# ==================== CLASSES AUXILIARES ====================
class RateLimiter:
    """Controla rate limit"""
    def __init__(self, max_requests=25, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = deque()
    
    def esperar_se_necessario(self):
        agora = time.time()
        while self.requests and agora - self.requests[0] > self.window:
            self.requests.popleft()
        
        if len(self.requests) >= self.max_requests:
            tempo_espera = self.window - (agora - self.requests[0]) + 1
            if tempo_espera > 0:
                time.sleep(tempo_espera)
        
        self.requests.append(time.time())

class EstatisticasGlobais:
    """Rastreia estatísticas"""
    def __init__(self):
        self.tentativas = defaultdict(int)
        self.sucessos = defaultdict(int)
        self.falhas = defaultdict(lambda: defaultdict(int))
        self.tempos = defaultdict(list)
    
    def registrar(self, modelo, sucesso, tempo, erro=None):
        self.tentativas[modelo] += 1
        if sucesso:
            self.sucessos[modelo] += 1
            self.tempos[modelo].append(tempo)
        else:
            motivo = self._classificar_erro(erro)
            self.falhas[modelo][motivo] += 1
    
    def _classificar_erro(self, erro):
        if not erro:
            return 'Desconhecido'
        erro = str(erro)
        if '503' in erro:
            return 'HTTP 503'
        elif '429' in erro:
            return 'Rate Limit'
        elif 'timeout' in erro.lower():
            return 'Timeout'
        return f'Outro ({erro[:30]}...)'
    
    def get_taxa_sucesso(self, modelo):
        if self.tentativas[modelo] == 0:
            return 0.0
        return (self.sucessos[modelo] / self.tentativas[modelo]) * 100
    
    def get_tempo_medio(self, modelo):
        if not self.tempos[modelo]:
            return 0.0
        return sum(self.tempos[modelo]) / len(self.tempos[modelo])

# ==================== FUNÇÕES API ====================
def chamar_groq(img_bytes, prompt, modelo_id, api_key):
    """Chama Groq API"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    payload = {
        "model": modelo_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        }],
        "max_completion_tokens": 4000,
        "temperature": 0.1
    }
    
    try:
        inicio = time.time()
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=90
        )
        tempo = time.time() - inicio
        
        if response.status_code == 200:
            data = response.json()
            return {
                'sucesso': True,
                'texto': data['choices'][0]['message']['content'],
                'tempo': tempo,
                'erro': None
            }
        else:
            return {
                'sucesso': False,
                'texto': None,
                'tempo': tempo,
                'erro': f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {'sucesso': False, 'texto': None, 'tempo': 0, 'erro': str(e)}

def chamar_gemini(img_bytes, prompt, modelo_id, api_key):
    """Chama Gemini API"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_id}:generateContent?key={api_key}"
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_base64}}
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4000}
    }
    
    try:
        inicio = time.time()
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=120)
        tempo = time.time() - inicio
        
        if response.status_code == 200:
            data = response.json()
            return {
                'sucesso': True,
                'texto': data['candidates'][0]['content']['parts'][0]['text'],
                'tempo': tempo,
                'erro': None
            }
        else:
            return {'sucesso': False, 'texto': None, 'tempo': tempo, 'erro': f"HTTP {response.status_code}"}
    except Exception as e:
        return {'sucesso': False, 'texto': None, 'tempo': 0, 'erro': str(e)}

# ==================== ANÁLISE DE COMPLEXIDADE ====================
def analisar_complexidade(img):
    """Calcula score de complexidade (0-100)"""
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    
    stats = ImageStat.Stat(img_gray)
    contraste = stats.stddev[0]
    
    gradiente_x = np.abs(np.diff(img_array, axis=1))
    gradiente_y = np.abs(np.diff(img_array, axis=0))
    nitidez = (gradiente_x.mean() + gradiente_y.mean()) / 2
    
    largura, altura = img.size
    megapixels = (largura * altura) / 1_000_000
    
    score = 0
    if contraste >= 80: score += 30
    elif contraste >= 60: score += 20
    elif contraste >= 40: score += 10
    
    if nitidez >= 15: score += 25
    elif nitidez >= 10: score += 15
    elif nitidez >= 5: score += 5
    
    if megapixels >= 4: score += 20
    elif megapixels >= 2: score += 15
    elif megapixels >= 1: score += 10
    else: score += 5
    
    return min(score, 100)

# ==================== PROCESSAMENTO ====================
def processar_pagina(img_bytes, score, api_keys, stats, progress_callback=None):
    """Processa uma página com fallback"""
    
    # Determinar estratégia
    if score >= 80:
        estrategia = Config.ESTRATEGIAS['muito_complexa']
    elif score >= 50:
        estrategia = Config.ESTRATEGIAS['media']
    else:
        estrategia = Config.ESTRATEGIAS['simples']
    
    prompt = PROMPT_COMPLEXO if score >= 70 else PROMPT_SIMPLES
    
    # Executar sequência de fallback
    for i, etapa in enumerate(estrategia['sequencia'], 1):
        modelo_key = etapa['modelo']
        modelo = Config.MODELOS[modelo_key]
        max_retries = etapa['retries']
        
        if progress_callback:
            progress_callback(f"Tentando {modelo['nome']}...")
        
        # Retry loop
        for tentativa in range(1, max_retries + 1):
            # Chamar API
            if modelo['tipo'] == 'groq':
                resultado = chamar_groq(img_bytes, prompt, modelo['id'], api_keys.get('groq'))
            else:
                resultado = chamar_gemini(img_bytes, prompt, modelo['id'], api_keys.get('gemini'))
            
            # Registrar
            stats.registrar(modelo_key, resultado['sucesso'], resultado['tempo'], resultado['erro'])
            
            if resultado['sucesso']:
                return {
                    'sucesso': True,
                    'texto': resultado['texto'],
                    'modelo': modelo['nome'],
                    'tempo': resultado['tempo'],
                    'tentativa': tentativa
                }
            
            # Retry com backoff
            if tentativa < max_retries and ('503' in str(resultado['erro']) or '429' in str(resultado['erro'])):
                espera = min(5 * (2 ** (tentativa - 1)), 60)
                time.sleep(espera)
    
    # Falha total
    raise Exception(f"Todas as tentativas falharam (score={score})")

def processar_pdf(pdf_bytes, pdf_name, api_keys):
    """Processa PDF completo"""
    
    # Converter para imagens
    with st.spinner("🔄 Convertendo PDF para imagens..."):
        images = convert_from_bytes(pdf_bytes, dpi=Config.DPI)
    
    st.success(f"✅ {len(images)} páginas convertidas")
    
    # Processar páginas
    stats = EstatisticasGlobais()
    resultados = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, img in enumerate(images, 1):
        try:
            # Converter para bytes
            buf = BytesIO()
            img.save(buf, format='PNG')
            img_bytes = buf.getvalue()
            
            # Analisar complexidade
            score = analisar_complexidade(img)
            
            # Atualizar UI
            status_text.text(f"📄 Processando página {i}/{len(images)} (Score: {score})")
            
            # Processar
            def update_status(msg):
                status_text.text(f"📄 Página {i}/{len(images)}: {msg}")
            
            resultado = processar_pagina(img_bytes, score, api_keys, stats, update_status)
            
            resultados.append({
                'pagina': i,
                'score': score,
                'modelo': resultado['modelo'],
                'tempo': resultado['tempo'],
                'texto': resultado['texto']
            })
            
            progress_bar.progress(i / len(images))
            
        except Exception as e:
            st.error(f"❌ Erro na página {i}: {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return resultados, stats

# ==================== INTERFACE STREAMLIT ====================
def main():
    st.title("📄 Sistema de OCR Híbrido Pro")
    st.markdown("**Versão 5.0** - Processamento inteligente com fallback em cascata")
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        st.subheader("🔑 API Keys")
        gemini_key = st.text_input("Gemini API Key", type="password", help="Obtenha em ai.google.dev")
        groq_key = st.text_input("Groq API Key", type="password", help="Obtenha em console.groq.com")
        
        st.divider()
        
        st.subheader("📊 Estratégias")
        st.markdown("""
        **Score ≥ 80:** Gemini prioritário  
        **Score 50-79:** Gemini → Groq  
        **Score < 50:** Groq prioritário
        """)
        
        st.divider()
        
        st.subheader("ℹ️ Sobre")
        st.markdown("""
        Sistema híbrido com:
        - 4 modelos de IA
        - Retry automático
        - Fallback inteligente
        - Estatísticas detalhadas
        """)
    
    # Área principal
    if not gemini_key or not groq_key:
        st.warning("⚠️ Configure as API Keys na barra lateral")
        st.stop()
    
    api_keys = {'gemini': gemini_key, 'groq': groq_key}
    
    # Upload
    st.header("📤 Upload de PDF")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo PDF",
        type=['pdf'],
        help=f"Tamanho máximo: {Config.MAX_FILE_SIZE}MB"
    )
    
    if uploaded_file:
        # Validar tamanho
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size > Config.MAX_FILE_SIZE:
            st.error(f"❌ Arquivo muito grande ({file_size:.1f}MB). Limite: {Config.MAX_FILE_SIZE}MB")
            st.stop()
        
        st.success(f"✅ {uploaded_file.name} ({file_size:.1f}MB)")
        
        # Botão processar
        if st.button("🚀 Processar PDF", type="primary", use_container_width=True):
            try:
                inicio = time.time()
                
                # Processar
                resultados, stats = processar_pdf(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    api_keys
                )
                
                tempo_total = time.time() - inicio
                
                # Resultados
                st.success(f"✅ Processamento concluído em {tempo_total/60:.1f} minutos!")
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["📊 Estatísticas", "📄 Resultados", "💾 Download"])
                
                with tab1:
                    st.header("📊 Estatísticas de Processamento")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Páginas Processadas", len(resultados))
                    with col2:
                        st.metric("Tempo Médio/Página", f"{tempo_total/len(resultados):.1f}s")
                    with col3:
                        taxa_geral = (sum(stats.sucessos.values()) / sum(stats.tentativas.values()) * 100) if sum(stats.tentativas.values()) > 0 else 0
                        st.metric("Taxa de Sucesso", f"{taxa_geral:.1f}%")
                    
                    st.subheader("Por Modelo")
                    for modelo_key, modelo_info in Config.MODELOS.items():
                        if stats.tentativas[modelo_key] > 0:
                            with st.expander(f"🤖 {modelo_info['nome']}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Tentativas", stats.tentativas[modelo_key])
                                    st.metric("Sucessos", stats.sucessos[modelo_key])
                                with col2:
                                    st.metric("Taxa Sucesso", f"{stats.get_taxa_sucesso(modelo_key):.1f}%")
                                    st.metric("Tempo Médio", f"{stats.get_tempo_medio(modelo_key):.2f}s")
                                
                                if stats.falhas[modelo_key]:
                                    st.write("**Falhas:**")
                                    for motivo, count in stats.falhas[modelo_key].items():
                                        st.write(f"- {motivo}: {count}x")
                
                with tab2:
                    st.header("📄 Texto Extraído")
                    
                    for res in resultados:
                        with st.expander(f"Página {res['pagina']} - {res['modelo']} ({res['tempo']:.1f}s) [Score: {res['score']}]"):
                            st.markdown(res['texto'])
                
                with tab3:
                    st.header("💾 Downloads")
                    
                    # JSON completo
                    json_data = {
                        'metadata': {
                            'arquivo': uploaded_file.name,
                            'data_processamento': datetime.now().isoformat(),
                            'tempo_total': tempo_total,
                            'total_paginas': len(resultados)
                        },
                        'resultados': resultados,
                        'estatisticas': {
                            'por_modelo': {
                                k: {
                                    'tentativas': stats.tentativas[k],
                                    'sucessos': stats.sucessos[k],
                                    'taxa_sucesso': stats.get_taxa_sucesso(k),
                                    'tempo_medio': stats.get_tempo_medio(k)
                                }
                                for k in Config.MODELOS.keys() if stats.tentativas[k] > 0
                            }
                        }
                    }
                    
                    st.download_button(
                        "📥 Baixar JSON Completo",
                        data=json.dumps(json_data, indent=2, ensure_ascii=False),
                        file_name=f"{uploaded_file.name}_ocr.json",
                        mime="application/json"
                    )
                    
                    # TXT simples
                    texto_completo = "\n\n".join([
                        f"=== PÁGINA {r['pagina']} ===\n{r['texto']}"
                        for r in resultados
                    ])
                    
                    st.download_button(
                        "📥 Baixar TXT Simples",
                        data=texto_completo,
                        file_name=f"{uploaded_file.name}_ocr.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Erro durante processamento: {e}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
