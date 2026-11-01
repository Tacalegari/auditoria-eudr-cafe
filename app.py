import os
import io
import re
import json
import zipfile
import hashlib
from datetime import datetime, timedelta

import requests
import streamlit as st
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import shape
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


st.set_page_config(
    page_title="Governança Geoespacial EUDR | Café",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------------------------------------------------------
# 1. AUTENTICAÇÃO COM CONTA DE SERVIÇO (STREAMLIT SECRETS)
# ------------------------------------------------------------------------------
@st.cache_resource
def inicializar_gee():
    try:
        if "gcp_service_account" in st.secrets:
            chave = dict(st.secrets["gcp_service_account"])
            credenciais = ee.ServiceAccountCredentials(
                chave["client_email"],
                key_data=chave["private_key"],
            )
            ee.Initialize(
                credenciais,
                project=chave.get("project_id", "cogent-script-449412-s4"),
            )
        else:
            ee.Initialize(project="cogent-script-449412-s4")
        return True, "Google Earth Engine conectado."
    except Exception as erro:
        return False, f"Falha ao conectar ao Google Earth Engine: {erro}"


gee_online, mensagem_gee = inicializar_gee()


# ------------------------------------------------------------------------------
# 2. BASE CADASTRAL LOCAL DE REFERÊNCIA
# ------------------------------------------------------------------------------
@st.cache_data
def carregar_base_referencia():
    caminho = "amostras_reais_50_propriedades.csv"
    if not os.path.exists(caminho):
        return None

    df = pd.read_csv(caminho, sep=";", dtype={"cod_imovel": "string"})
    return df


df_props_ref = carregar_base_referencia()


def normalizar_car(valor):
    if valor is None:
        return ""
    return re.sub(r"\s+", "", str(valor)).strip().upper()


def obter_url_wfs():
    try:
        return str(st.secrets.get("SICAR_WFS_URL", "")).strip()
    except Exception:
        return ""


# ------------------------------------------------------------------------------
# 3. BUSCA DE POLÍGONOS: BASE LOCAL E SICAR/WFS CONFIGURADO
# ------------------------------------------------------------------------------
def consultar_sicar_nacional(cod_car):
    cod_car = normalizar_car(cod_car)

    if not cod_car:
        raise ValueError("O código do CAR não foi informado.")

    if df_props_ref is not None:
        colunas_obrigatorias = {
            "cod_imovel",
            "geometry",
            "municipio",
            "cod_estado",
            "num_area",
        }
        faltantes = colunas_obrigatorias.difference(df_props_ref.columns)
        if faltantes:
            raise ValueError(
                "A base local não contém as colunas obrigatórias: "
                + ", ".join(sorted(faltantes))
            )

        codigos_normalizados = (
            df_props_ref["cod_imovel"]
            .fillna("")
            .astype(str)
            .map(normalizar_car)
        )
        match = df_props_ref.loc[codigos_normalizados == cod_car]

        if len(match) > 1:
            raise ValueError(
                f"O CAR {cod_car} aparece mais de uma vez na base de referência."
            )

        if len(match) == 1:
            registro = match.iloc[0]
            try:
                geom = wkt.loads(str(registro["geometry"])).buffer(0)
            except Exception as erro:
                raise ValueError(
                    f"Geometria inválida para o CAR {cod_car}: {erro}"
                ) from erro

            if geom.is_empty or not geom.is_valid:
                raise ValueError(
                    f"A geometria do CAR {cod_car} está vazia ou inválida."
                )

            try:
                area_ha = float(str(registro["num_area"]).replace(",", "."))
            except Exception as erro:
                raise ValueError(
                    f"Área inválida para o CAR {cod_car}: {registro['num_area']}"
                ) from erro

            codigo_oficial = normalizar_car(registro["cod_imovel"])
            mun_uf = f"{registro['municipio']} - {registro['cod_estado']}"
            return geom, codigo_oficial, mun_uf, area_ha, "BASE_LOCAL"

    url_wfs = obter_url_wfs()
    if not url_wfs:
        raise LookupError(
            f"O CAR {cod_car} não foi encontrado na base local e o segredo "
            "SICAR_WFS_URL não está configurado."
        )

    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": "sicar:sicar_imoveis_poligono",
        "outputFormat": "application/json",
        "cql_filter": f"cod_imovel='{cod_car}'",
        "maxFeatures": 2,
    }

    try:
        resposta = requests.get(url_wfs, params=params, timeout=30)
        resposta.raise_for_status()
        dados = resposta.json()
    except requests.RequestException as erro:
        raise RuntimeError(
            f"Falha ao consultar o serviço WFS para o CAR {cod_car}: {erro}"
        ) from erro
    except ValueError as erro:
        raise RuntimeError(
            f"O WFS retornou uma resposta JSON inválida para o CAR {cod_car}."
        ) from erro

    feicoes = dados.get("features", [])
    if not feicoes:
        raise LookupError(f"O CAR {cod_car} não foi encontrado no serviço WFS.")
    if len(feicoes) > 1:
        raise LookupError(
            f"A consulta retornou mais de um imóvel para o CAR {cod_car}."
        )

    feicao = feicoes[0]
    propriedades = feicao.get("properties", {})

    try:
        geom = shape(feicao["geometry"]).buffer(0)
    except Exception as erro:
        raise ValueError(
            f"O WFS retornou geometria inválida para o CAR {cod_car}: {erro}"
        ) from erro

    if geom.is_empty or not geom.is_valid:
        raise ValueError(
            f"O WFS retornou geometria vazia ou inválida para o CAR {cod_car}."
        )

    codigo_retornado = normalizar_car(
        propriedades.get("cod_imovel", propriedades.get("cod_car", cod_car))
    )
    if codigo_retornado != cod_car:
        raise ValueError(
            f"O WFS retornou o código {codigo_retornado}, diferente do código "
            f"pesquisado {cod_car}."
        )

    area_valor = propriedades.get("num_area", propriedades.get("val_area"))
    if area_valor in (None, ""):
        raise ValueError(f"O WFS não retornou a área do imóvel {cod_car}.")

    try:
        area_ha = float(str(area_valor).replace(",", "."))
    except Exception as erro:
        raise ValueError(
            f"O WFS retornou área inválida para o CAR {cod_car}: {area_valor}"
        ) from erro

    municipio = propriedades.get(
        "nom_municipio",
        propriedades.get("municipio", "Município não informado"),
    )
    uf = propriedades.get(
        "sig_uf",
        propriedades.get("cod_estado", "UF não informada"),
    )

    return geom, cod_car, f"{municipio} - {uf}", area_ha, "SICAR_WFS"


# ------------------------------------------------------------------------------
# 4. MOTOR ESPECTRAL SENTINEL-2
# ------------------------------------------------------------------------------
def validar_intervalo_datas(data_iso):
    try:
        data_final = datetime.strptime(data_iso, "%Y-%m-%d").date()
    except ValueError as erro:
        raise ValueError("A data final deve estar no formato AAAA-MM-DD.") from erro

    if data_final <= datetime.strptime("2021-01-01", "%Y-%m-%d").date():
        raise ValueError("A data final precisa ser posterior a 01/01/2021.")

    # filterDate usa limite final exclusivo. Somamos um dia para incluir a data atual.
    return (data_final + timedelta(days=1)).strftime("%Y-%m-%d")


def analisar_espectro_satelite(geom_shapely, data_iso, cod_car=""):
    cod_car = normalizar_car(cod_car)

    if not gee_online:
        raise RuntimeError(mensagem_gee)
    if geom_shapely is None or geom_shapely.is_empty or not geom_shapely.is_valid:
        raise ValueError(f"Geometria ausente ou inválida para o CAR {cod_car}.")

    data_final_exclusiva = validar_intervalo_datas(data_iso)

    try:
        geom_gee = ee.Geometry(geom_shapely.__geo_interface__)

        def mascarar_e_calcular_ndvi(img):
            qa = img.select("QA60")
            mascara_nuvem = qa.bitwiseAnd(1 << 10).eq(0)
            mascara_cirrus = qa.bitwiseAnd(1 << 11).eq(0)
            mascara = mascara_nuvem.And(mascara_cirrus)

            reflectancia = img.select(["B4", "B8"]).divide(10000)
            ndvi = reflectancia.normalizedDifference(["B8", "B4"]).rename("NDVI")

            return (
                img.addBands(reflectancia, overwrite=True)
                .addBands(ndvi)
                .updateMask(mascara)
                .copyProperties(img, ["system:time_start"])
            )

        colecao_base = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom_gee)
            .filterDate("2020-01-01", "2021-01-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
            .map(mascarar_e_calcular_ndvi)
        )

        colecao_pos = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom_gee)
            .filterDate("2021-01-01", data_final_exclusiva)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
            .map(mascarar_e_calcular_ndvi)
        )

        inicio_recente = "2024-01-01"
        colecao_recente = colecao_pos.filterDate(
            inicio_recente,
            data_final_exclusiva,
        )

        qtd_base = int(colecao_base.size().getInfo())
        qtd_pos = int(colecao_pos.size().getInfo())
        qtd_recente = int(colecao_recente.size().getInfo())

        if qtd_base == 0:
            raise RuntimeError(
                f"Nenhuma imagem Sentinel-2 válida em 2020 para o CAR {cod_car}."
            )
        if qtd_pos == 0:
            raise RuntimeError(
                f"Nenhuma imagem Sentinel-2 válida pós-2020 para o CAR {cod_car}."
            )
        if qtd_recente == 0:
            raise RuntimeError(
                f"Nenhuma imagem Sentinel-2 válida desde 2024 para o CAR {cod_car}."
            )

        parametros = {
            "geometry": geom_gee,
            "scale": 20,
            "maxPixels": 1e8,
            "bestEffort": True,
        }

        v_base = (
            colecao_base.select("NDVI")
            .median()
            .reduceRegion(reducer=ee.Reducer.mean(), **parametros)
            .get("NDVI")
            .getInfo()
        )

        v_min = (
            colecao_pos.select("NDVI")
            .reduce(ee.Reducer.percentile([10]))
            .reduceRegion(reducer=ee.Reducer.mean(), **parametros)
            .get("NDVI_p10")
            .getInfo()
        )

        v_rec = (
            colecao_recente.select("NDVI")
            .median()
            .reduceRegion(reducer=ee.Reducer.mean(), **parametros)
            .get("NDVI")
            .getInfo()
        )

        if any(valor is None for valor in (v_base, v_min, v_rec)):
            raise RuntimeError(
                f"O Earth Engine não retornou NDVI válido para o CAR {cod_car}."
            )

        v_base = round(float(v_base), 3)
        v_min = round(float(v_min), 3)
        v_rec = round(float(v_rec), 3)

    except Exception as erro:
        if isinstance(erro, (RuntimeError, ValueError)):
            raise
        raise RuntimeError(
            f"Erro na análise Sentinel-2 do CAR {cod_car}: {erro}"
        ) from erro

    if v_base >= 0.50:
        if v_min < 0.35 and v_rec >= 0.45:
            status = "Conforme - Poda Agronômica Mitigada (Alerta JRC Cancelado)"
            parecer = "LIBERADO"
        elif v_min < 0.35 and v_rec < 0.40:
            status = "Alerta - Supressão sem Rebrota Pós-2020 Detectada"
            parecer = "RETIDO"
        else:
            status = "Conforme - Uso Consolidado Regular"
            parecer = "LIBERADO"
    else:
        status = "Em Análise - Cobertura Histórica Baixa"
        parecer = "ANÁLISE ADICIONAL"

    diagnostico = {
        "qtd_imagens_base": qtd_base,
        "qtd_imagens_pos": qtd_pos,
        "qtd_imagens_recente": qtd_recente,
        "metodo_minimo": "Percentil 10 do NDVI pós-2020",
    }

    return v_base, v_min, v_rec, status, parecer, diagnostico


def plotar_curva_fenologica(cod_car, v_base, v_min, v_rec, data_iso, parecer):
    datas = pd.date_range(start="2020-01-01", end=data_iso, freq="MS")
    serie = []

    if parecer == "RETIDO":
        for data in datas:
            if data.year == 2020:
                valor = v_base + 0.02 * np.sin(2 * np.pi * data.month / 12)
            elif data.year == 2021:
                progresso = ((data - pd.Timestamp("2021-01-01")).days) / 365
                valor = v_base - (v_base - v_min) * min(max(progresso, 0), 1)
            else:
                valor = v_min + 0.015 * np.sin(2 * np.pi * data.month / 12)
            serie.append(valor)
        cor_linha = "#d32f2f"
    else:
        for data in datas:
            if data.year == 2020:
                valor = v_base + 0.03 * np.sin(2 * np.pi * data.month / 12)
            elif data.year in (2021, 2022):
                progresso = ((data - pd.Timestamp("2021-01-01")).days) / (2 * 365)
                valor = v_base - (v_base - v_min) * min(max(progresso, 0), 1)
            else:
                progresso = ((data - pd.Timestamp("2023-01-01")).days) / (3.5 * 365)
                valor = (
                    v_min
                    + (v_rec - v_min) * min(max(progresso, 0), 1)
                    + 0.03 * np.sin(2 * np.pi * data.month / 12)
                )
            serie.append(valor)
        cor_linha = "#1b5e20"

    # A curva é uma representação fenológica baseada nos três indicadores agregados.
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(datas, serie, color=cor_linha, linewidth=2.2, label="Perfil fenológico estimado")
    ax.axvline(
        pd.to_datetime("2020-12-31"),
        color="#b71c1c",
        linestyle="--",
        linewidth=1.5,
        label="Marco temporal EUDR (31/12/2020)",
    )
    ax.axhline(0.35, color="#f57f17", linestyle=":", label="Limiar NDVI 0,35")
    ax.fill_between(
        datas,
        0.35,
        0.85,
        color="#ffebee" if parecer == "RETIDO" else "#e8f5e9",
        alpha=0.4,
        label="Faixa de referência",
    )
    ax.set_title(
        f"Indicadores Sentinel-2 | {cod_car[:28]}...",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_ylim(0.15, 0.85)
    ax.set_ylabel("NDVI", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()

    buffer_imagem = io.BytesIO()
    fig.savefig(buffer_imagem, format="png", dpi=220)
    plt.close(fig)
    buffer_imagem.seek(0)
    return buffer_imagem


# ------------------------------------------------------------------------------
# 5. GERADOR DO LAUDO EM PDF
# ------------------------------------------------------------------------------
def emitir_pdf_laudo(
    cod_car,
    mun_uf,
    area_ha,
    v_base,
    v_min,
    v_rec,
    status,
    graf_buf,
    data_str,
    geom,
    parecer,
    fonte_geom,
    diagnostico,
):
    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf_pdf,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    cor_primaria = colors.HexColor("#b71c1c") if parecer == "RETIDO" else colors.HexColor("#1b5e20")
    cor_secundaria = colors.HexColor("#c62828") if parecer == "RETIDO" else colors.HexColor("#2e7d32")
    cor_fundo = colors.HexColor("#ffebee") if parecer == "RETIDO" else colors.HexColor("#f1f8e9")

    style_title = ParagraphStyle(
        "TP", parent=styles["Normal"], fontSize=13, leading=16,
        fontName="Helvetica-Bold", textColor=cor_primaria,
    )
    style_h2 = ParagraphStyle(
        "H2", parent=styles["Normal"], fontSize=10, leading=13,
        fontName="Helvetica-Bold", textColor=cor_secundaria,
    )
    style_th = ParagraphStyle(
        "TH", parent=styles["Normal"], fontSize=8.5, leading=11,
        fontName="Helvetica-Bold", textColor=colors.whitesmoke,
    )
    style_td = ParagraphStyle("TD", parent=styles["Normal"], fontSize=8, leading=11)
    style_body = ParagraphStyle(
        "BD", parent=styles["Normal"], fontSize=8, leading=11, alignment=4,
    )

    centroide = geom.centroid
    lat_txt = f"{centroide.y:.5f}°"
    lon_txt = f"{centroide.x:.5f}°"

    dados_hash = (
        f"{cod_car}_{mun_uf}_{area_ha}_{lat_txt}_{lon_txt}_{status}_"
        f"{data_str}_{fonte_geom}_{v_base}_{v_min}_{v_rec}"
    ).encode("utf-8")
    sha256 = hashlib.sha256(dados_hash).hexdigest()

    texto_regularidade = (
        f"<b>PARECER DE NÃO-CONFORMIDADE: {status}</b>. "
        "O indicador espectral atende ao critério interno de retenção."
        if parecer == "RETIDO"
        else f"Parecer automatizado: <b>{status}</b>."
    )

    dados_intro = [
        [Paragraph("Item", style_th), Paragraph("Detalhes", style_th)],
        [Paragraph("<b>A. Propósito</b>", style_td), Paragraph("Apoio técnico à devida diligência EUDR por meio de geometria cadastrada e indicadores Sentinel-2.", style_td)],
        [Paragraph("<b>B. Identificação do Imóvel</b>", style_td), Paragraph(
            f"<b>Código do CAR:</b> {cod_car}<br/>"
            f"<b>Município/UF:</b> {mun_uf}<br/>"
            f"<b>Área registrada:</b> {area_ha}<br/>"
            f"<b>Centróide:</b> Lat {lat_txt} | Long {lon_txt}<br/>"
            f"<b>Fonte da geometria:</b> {fonte_geom}", style_td)],
        [Paragraph("<b>C. Parecer</b>", style_td), Paragraph(texto_regularidade, style_td)],
        [Paragraph("<b>D. Indicadores</b>", style_td), Paragraph(
            f"NDVI Base 2020: {v_base:.3f} | Percentil 10 pós-2020: {v_min:.3f} | "
            f"NDVI recente: {v_rec:.3f}.<br/>"
            f"Imagens válidas: base={diagnostico['qtd_imagens_base']}, "
            f"pós-2020={diagnostico['qtd_imagens_pos']}, "
            f"recentes={diagnostico['qtd_imagens_recente']}.", style_td)],
        [Paragraph("<b>E. Autenticidade</b>", style_td), Paragraph(
            f"<b>Hash SHA-256:</b> <font size=5>{sha256}</font><br/>"
            f"<b>Data da análise:</b> {data_str}", style_td)],
    ]

    tabela = Table(dados_intro, colWidths=[115, 405])
    tabela.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), cor_secundaria),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 1), (0, -1), cor_fundo),
    ]))

    elementos = [
        Paragraph("Relatório de Verificação Geoespacial EUDR", style_title),
        Spacer(1, 10),
        tabela,
        Spacer(1, 10),
        Paragraph("Metodologia", style_h2),
        Spacer(1, 4),
        Paragraph(
            "A geometria é obtida por correspondência exata do código do CAR. "
            "A análise espectral utiliza a coleção COPERNICUS/S2_SR_HARMONIZED, "
            "máscara QA60, mediana de 2020, percentil 10 pós-2020 e mediana recente. "
            "Não são utilizados polígonos ou valores NDVI substitutos quando uma consulta falha.",
            style_body,
        ),
        Spacer(1, 10),
        Paragraph("Evidência gráfica", style_h2),
        Spacer(1, 4),
        ReportLabImage(graf_buf, width=520, height=200),
        Spacer(1, 8),
        Paragraph("Conclusão", style_h2),
        Spacer(1, 4),
    ]

    if parecer == "RETIDO":
        conclusao = (
            "Os indicadores automatizados atingiram o critério interno de retenção. "
            "O resultado deve ser submetido à validação geoespacial e documental antes "
            "de qualquer decisão comercial definitiva."
        )
    elif parecer == "ANÁLISE ADICIONAL":
        conclusao = (
            "A cobertura histórica apresentou baixa resposta espectral. É necessária "
            "verificação complementar com imagens, documentos e verdade de campo."
        )
    else:
        conclusao = (
            "Os indicadores automatizados não atingiram o critério interno de retenção. "
            "O resultado apoia a devida diligência, mas não substitui validação jurídica, "
            "documental ou interpretação técnica independente."
        )

    elementos.append(Paragraph(conclusao, style_body))
    doc.build(elementos)
    buf_pdf.seek(0)
    return buf_pdf


def criar_resultado(geom, cod_car, mun_uf, area_ha, fonte_geom, data_iso):
    v_base, v_min, v_rec, status, parecer, diagnostico = analisar_espectro_satelite(
        geom, data_iso, cod_car=cod_car
    )
    grafico = plotar_curva_fenologica(
        cod_car, v_base, v_min, v_rec, data_iso, parecer
    )
    return {
        "cod_id": cod_car,
        "mun_uf": mun_uf,
        "area_ha": area_ha,
        "v_base": v_base,
        "v_min": v_min,
        "v_rec": v_rec,
        "status": status,
        "parecer": parecer,
        "fonte_geom": fonte_geom,
        "diagnostico": diagnostico,
        "grafico": grafico,
    }


# ------------------------------------------------------------------------------
# 6. INTERFACE STREAMLIT
# ------------------------------------------------------------------------------
st.title("☕ Plataforma de Governança Geoespacial — Compliance EUDR")
st.caption("Protocolo automatizado de apoio à análise e devida diligência na cafeicultura")

if gee_online:
    st.success("Google Earth Engine conectado.")
else:
    st.error(mensagem_gee)

if df_props_ref is None and not obter_url_wfs():
    st.warning(
        "Nenhuma base local foi carregada e SICAR_WFS_URL não está configurado. "
        "As consultas por CAR serão bloqueadas para evitar resultados simulados."
    )

st.markdown("---")

data_iso = datetime.now().strftime("%Y-%m-%d")
data_formatada = datetime.now().strftime("%d/%m/%Y")

tab_ind, tab_lot, tab_vet, tab_esg = st.tabs([
    "🔍 Análise Individual",
    "📦 Análise em Lote",
    "📂 Upload de Vetores (GeoJSON)",
    "📊 Dashboard ESG & Validação TCC",
])

if "historico_analises" not in st.session_state:
    st.session_state.historico_analises = []


with tab_ind:
    st.subheader("Consulta e Emissão de Laudo Individual")
    car_entrada = st.text_input(
        "Código completo do CAR:",
        placeholder="Ex.: MG-3101607-...",
    )

    if st.button("🔍 Analisar Imóvel", type="primary", use_container_width=True):
        if not car_entrada.strip():
            st.warning("Digite o código completo do CAR.")
        else:
            try:
                with st.spinner("1/2. Consultando a geometria do imóvel..."):
                    geom, cod_car, mun_uf, area_ha, fonte_geom = consultar_sicar_nacional(car_entrada)

                with st.spinner("2/2. Calculando indicadores Sentinel-2..."):
                    resultado = criar_resultado(
                        geom, cod_car, mun_uf, area_ha, fonte_geom, data_iso
                    )
                    pdf_buf = emitir_pdf_laudo(
                        cod_car=cod_car,
                        mun_uf=mun_uf,
                        area_ha=f"{area_ha:.2f}".replace(".", ",") + " ha",
                        v_base=resultado["v_base"],
                        v_min=resultado["v_min"],
                        v_rec=resultado["v_rec"],
                        status=resultado["status"],
                        graf_buf=resultado["grafico"],
                        data_str=data_formatada,
                        geom=geom,
                        parecer=resultado["parecer"],
                        fonte_geom=fonte_geom,
                        diagnostico=resultado["diagnostico"],
                    )

                historico = {k: v for k, v in resultado.items() if k not in ("grafico", "diagnostico")}
                historico.update(resultado["diagnostico"])
                st.session_state.historico_analises.append(historico)

                st.markdown("### Parecer de Conformidade Socioambiental")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Localização", mun_uf)
                m2.metric("Área analisada", f"{area_ha:.2f} ha")
                m3.metric("NDVI Base 2020", f"{resultado['v_base']:.3f}")
                m4.metric("NDVI recente", f"{resultado['v_rec']:.3f}")
                st.caption(f"Fonte da geometria: {fonte_geom}")

                if resultado["parecer"] == "LIBERADO":
                    st.success(f"PARECER AUTOMATIZADO: {resultado['status']}")
                elif resultado["parecer"] == "ANÁLISE ADICIONAL":
                    st.warning(f"PARECER AUTOMATIZADO: {resultado['status']}")
                else:
                    st.error(f"PARECER AUTOMATIZADO: {resultado['status']}")

                col_g, col_p = st.columns([2, 1])
                with col_g:
                    st.image(
                        resultado["grafico"],
                        caption="Representação dos indicadores NDVI",
                        use_container_width=True,
                    )
                with col_p:
                    st.markdown("#### Relatório técnico")
                    st.download_button(
                        "📄 Baixar relatório (.PDF)",
                        data=pdf_buf,
                        file_name=f"Laudo_EUDR_{cod_car[:20]}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

                with st.expander("Diagnóstico da análise"):
                    st.json(resultado["diagnostico"])

            except Exception as erro:
                st.error(f"Não foi possível concluir a análise: {erro}")


with tab_lot:
    st.subheader("Análise Coletiva de Fornecedores")
    texto_lote = st.text_area(
        "Códigos completos do CAR, separados por ponto e vírgula ou linha:",
        height=130,
    )
    arq_lote = st.file_uploader("Ou envie um arquivo .TXT:", type=["txt"])

    conteudo_lote = texto_lote
    if not conteudo_lote.strip() and arq_lote is not None:
        conteudo_lote = arq_lote.getvalue().decode("utf-8-sig")

    itens = re.split(r"[;\n,\s]+", conteudo_lote.strip()) if conteudo_lote.strip() else []
    lista_cars = list(dict.fromkeys(normalizar_car(item) for item in itens if len(normalizar_car(item)) > 8))

    if lista_cars:
        st.info(f"{len(lista_cars)} código(s) único(s) identificado(s).")

    if st.button("🚀 Executar Análise do Lote", type="primary"):
        if not lista_cars:
            st.warning("Insira ao menos um código de CAR.")
        else:
            resultados = []
            erros = []
            zip_buffer = io.BytesIO()
            progresso = st.progress(0)
            status_execucao = st.empty()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for indice, codigo in enumerate(lista_cars):
                    status_execucao.text(
                        f"Processando {indice + 1}/{len(lista_cars)}: {codigo[:32]}..."
                    )
                    try:
                        geom, cid, mun, area, fonte = consultar_sicar_nacional(codigo)
                        resultado = criar_resultado(geom, cid, mun, area, fonte, data_iso)
                        pdf = emitir_pdf_laudo(
                            cid, mun, f"{area:.2f}".replace(".", ",") + " ha",
                            resultado["v_base"], resultado["v_min"], resultado["v_rec"],
                            resultado["status"], resultado["grafico"], data_formatada,
                            geom, resultado["parecer"], fonte, resultado["diagnostico"],
                        )
                        zf.writestr(f"Laudo_EUDR_{cid[:22]}.pdf", pdf.getvalue())

                        item = {k: v for k, v in resultado.items() if k not in ("grafico", "diagnostico")}
                        item.update(resultado["diagnostico"])
                        resultados.append(item)
                        st.session_state.historico_analises.append(item)
                    except Exception as erro:
                        erros.append({"cod_id": codigo, "erro": str(erro)})
                    finally:
                        progresso.progress((indice + 1) / len(lista_cars))

            status_execucao.empty()

            if resultados:
                st.success(f"{len(resultados)} propriedade(s) analisada(s) com sucesso.")
                df_resultados = pd.DataFrame(resultados)
                st.dataframe(df_resultados, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📊 Baixar planilha consolidada (.CSV)",
                        df_resultados.to_csv(index=False, sep=";").encode("utf-8-sig"),
                        f"analise_lote_{data_iso}.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                with col2:
                    zip_buffer.seek(0)
                    st.download_button(
                        "📦 Baixar laudos válidos (.ZIP)",
                        zip_buffer,
                        f"laudos_individuais_{data_iso}.zip",
                        "application/zip",
                        use_container_width=True,
                    )

            if erros:
                st.error(f"{len(erros)} CAR(s) não foram processados. Nenhum dado substituto foi usado.")
                df_erros = pd.DataFrame(erros)
                st.dataframe(df_erros, use_container_width=True)
                st.download_button(
                    "Baixar relatório de erros (.CSV)",
                    df_erros.to_csv(index=False, sep=";").encode("utf-8-sig"),
                    f"erros_lote_{data_iso}.csv",
                    "text/csv",
                )


with tab_vet:
    st.subheader("Análise de Mapas e Polígonos (GeoJSON)")
    st.write("O GeoJSON deve estar em coordenadas geográficas WGS84, longitude/latitude (EPSG:4326).")
    arq_up = st.file_uploader("Selecione o arquivo de talhões:", type=["geojson", "json"])

    if arq_up is not None:
        try:
            dados_geo = json.loads(arq_up.getvalue().decode("utf-8-sig"))
            feicoes = dados_geo.get("features", [])
            if not feicoes:
                raise ValueError("O arquivo não contém feições GeoJSON.")
            st.success(f"Polígonos identificados: {len(feicoes)}")

            if st.button("Analisar Polígonos Importados", type="primary"):
                progresso = st.progress(0)
                resultados_vet = []
                erros_vet = []

                for indice, feicao in enumerate(feicoes):
                    try:
                        geom = shape(feicao.get("geometry")).buffer(0)
                        if geom.is_empty or not geom.is_valid:
                            raise ValueError("Geometria vazia ou inválida.")

                        props = feicao.get("properties", {})
                        cid = normalizar_car(props.get("cod_imovel", f"POLIGONO_{indice + 1}"))
                        mun = str(props.get("municipio", "Talhão importado"))

                        if props.get("num_area") not in (None, ""):
                            area = float(str(props["num_area"]).replace(",", "."))
                        else:
                            raise ValueError("A propriedade num_area não foi informada.")

                        resultado = criar_resultado(
                            geom, cid, mun, area, "GEOJSON_USUARIO", data_iso
                        )
                        item = {k: v for k, v in resultado.items() if k not in ("grafico", "diagnostico")}
                        item.update(resultado["diagnostico"])
                        resultados_vet.append(item)
                        st.session_state.historico_analises.append(item)
                    except Exception as erro:
                        erros_vet.append({"feicao": indice + 1, "erro": str(erro)})
                    finally:
                        progresso.progress((indice + 1) / len(feicoes))

                if resultados_vet:
                    st.success(f"{len(resultados_vet)} polígono(s) analisado(s).")
                    st.dataframe(pd.DataFrame(resultados_vet), use_container_width=True)
                if erros_vet:
                    st.error(f"{len(erros_vet)} polígono(s) apresentaram erro.")
                    st.dataframe(pd.DataFrame(erros_vet), use_container_width=True)

        except Exception as erro:
            st.error(f"Erro ao processar o arquivo vetorial: {erro}")


with tab_esg:
    st.subheader("Relatório Geral e Exportação")

    if not st.session_state.historico_analises:
        st.info("Nenhuma análise válida foi realizada nesta sessão.")
    else:
        df_dash = pd.DataFrame(st.session_state.historico_analises).drop_duplicates(
            subset=["cod_id"], keep="last"
        )
        total = len(df_dash)
        area_total = pd.to_numeric(df_dash["area_ha"], errors="coerce").fillna(0).sum()
        area_liberada = pd.to_numeric(
            df_dash.loc[df_dash["parecer"] == "LIBERADO", "area_ha"],
            errors="coerce",
        ).fillna(0).sum()
        taxa_liberada = (df_dash["parecer"].eq("LIBERADO").sum() / total) * 100 if total else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Propriedades analisadas", str(total))
        k2.metric("Área mapeada", f"{area_total:,.1f} ha")
        k3.metric("Área com parecer liberado", f"{area_liberada:,.1f} ha")
        k4.metric("Taxa de parecer liberado", f"{taxa_liberada:.1f}%")

        st.markdown("---")
        st.markdown("#### Matriz de validação científica")
        st.info(
            "Acurácia, precisão, revocação, F1-score e taxa de falsos positivos "
            "somente são calculados quando existe a coluna independente verdade_campo."
        )

        if "verdade_campo" in df_dash.columns and df_dash["verdade_campo"].notna().any():
            verdade = df_dash["verdade_campo"].astype(str).str.upper().str.strip()
            previsto = df_dash["parecer"].astype(str).str.upper().str.strip()

            vp = int(((previsto == "RETIDO") & (verdade == "DESMATE")).sum())
            fp = int(((previsto == "RETIDO") & (verdade == "CAFEICULTURA")).sum())
            fn = int(((previsto == "LIBERADO") & (verdade == "DESMATE")).sum())
            vn = int(((previsto == "LIBERADO") & (verdade == "CAFEICULTURA")).sum())
            total_matriz = vp + fp + fn + vn

            acuracia = ((vp + vn) / total_matriz) * 100 if total_matriz else 0
            precisao = (vp / (vp + fp)) * 100 if (vp + fp) else 0
            revocacao = (vp / (vp + fn)) * 100 if (vp + fn) else 0
            f1 = (2 * precisao * revocacao / (precisao + revocacao)) if (precisao + revocacao) else 0
            taxa_fp = (fp / (fp + vn)) * 100 if (fp + vn) else 0

            st.table(pd.DataFrame([
                {"Classificação": "Retido", "Verdade: Desmate": f"{vp} (VP)", "Verdade: Cafeicultura": f"{fp} (FP)"},
                {"Classificação": "Liberado", "Verdade: Desmate": f"{fn} (FN)", "Verdade: Cafeicultura": f"{vn} (VN)"},
            ]))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Acurácia", f"{acuracia:.1f}%")
            m2.metric("Precisão", f"{precisao:.1f}%")
            m3.metric("F1-score", f"{f1:.1f}%")
            m4.metric("Taxa de FP", f"{taxa_fp:.2f}%")
        else:
            st.warning(
                "Métricas de validação não calculadas: falta verdade de campo independente."
            )

        st.markdown("---")
        col_graf, col_json = st.columns(2)

        with col_graf:
            st.markdown("#### Distribuição dos pareceres")
            contagem = df_dash["parecer"].value_counts()
            cores = {
                "LIBERADO": "#2e7d32",
                "ANÁLISE ADICIONAL": "#fbc02d",
                "RETIDO": "#d32f2f",
            }
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie(
                contagem,
                labels=contagem.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=[cores.get(chave, "#9e9e9e") for chave in contagem.index],
                wedgeprops={"width": 0.4, "edgecolor": "white"},
            )
            ax.axis("equal")
            st.pyplot(fig)
            plt.close(fig)

        with col_json:
            st.markdown("#### Estrutura JSON para integração")
            colunas_exportacao = [
                coluna for coluna in [
                    "cod_id", "mun_uf", "area_ha", "v_base", "v_min", "v_rec",
                    "parecer", "status", "fonte_geom", "qtd_imagens_base",
                    "qtd_imagens_pos", "qtd_imagens_recente",
                ] if coluna in df_dash.columns
            ]
            payload = {
                "ddsReference": f"DDS-BR-COFFEE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "commodity": "0901 - Coffee",
                "cutOffDate": "2020-12-31",
                "summary": {
                    "totalAudited": total,
                    "totalAreaHa": round(float(area_total), 2),
                    "releasedOpinionRate": f"{taxa_liberada:.1f}%",
                },
                "plots": df_dash[colunas_exportacao].to_dict(orient="records"),
            }
            json_str = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                "🌐 Baixar estrutura JSON",
                json_str,
                f"DDS_EUDR_{data_iso}.json",
                "application/json",
            )
            with st.expander("Visualizar JSON"):
                st.code(json_str, language="json")


st.markdown("---")
st.caption(
    "Ferramenta de apoio à governança EUDR. Resultados automatizados exigem "
    "validação técnica, documental e jurídica antes de decisões definitivas."
)
