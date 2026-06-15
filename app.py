import io, math
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MultipleLocator

st.set_page_config(page_title="Planilla de inspección de represas", layout="wide")
st.title("Planilla de inspección de represas")
st.write("Carga puntos topográficos, calcula parámetros geométricos y genera un informe PDF estilo planilla de inspección.")

uploaded = st.file_uploader("Subí tu archivo Excel/CSV/TXT", type=["csv", "txt", "xlsx"])

def read_file(file):
    name = file.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(file)
    if name.endswith(".txt"):
        content = file.getvalue().decode("utf-8", errors="ignore")
        try:
            return pd.read_csv(io.StringIO(content), sep=None, engine="python")
        except Exception:
            return pd.read_csv(io.StringIO(content), sep=r"\s+", header=None)
    return pd.read_csv(file, sep=None, engine="python")

def norm_text(x):
    return str(x).strip().lower()

def format_pk(m):
    m = float(m)
    km = int(m // 1000)
    rest = m - km * 1000
    return f"{km}+{rest:06.2f}"

def pk_formatter(x, pos):
    return format_pk(x)

def calcular_distancia_acumulada(df):
    out = df.copy().reset_index(drop=True)
    out["Distancia tramo"] = np.sqrt(out["X"].diff()**2 + out["Y"].diff()**2).fillna(0)
    out["Distancia acumulada"] = out["Distancia tramo"].cumsum()
    out["Progresiva"] = out["Distancia acumulada"].apply(format_pk)
    return out

def calcular_anchos_pares(base_df):
    base_df = base_df.copy()
    base_df["Numero_num"] = pd.to_numeric(base_df["Numero"], errors="coerce")
    base_df = base_df.sort_values("Numero_num").reset_index(drop=True)
    registros = []
    for i in range(0, len(base_df) - 1, 2):
        p1 = base_df.iloc[i]
        p2 = base_df.iloc[i + 1]
        H = math.sqrt((p2["X"] - p1["X"])**2 + (p2["Y"] - p1["Y"])**2)
        registros.append({
            "Base 1": p1["Numero"],
            "Base 2": p2["Numero"],
            "Cota Base 1": p1["Z"],
            "Cota Base 2": p2["Z"],
            "Ancho corona (m)": H,
            "Diferencia cota (m)": p2["Z"] - p1["Z"]
        })
    return pd.DataFrame(registros)

def create_mdt_figure(data, cmap="terrain", mostrar_curvas=True, intervalo=0.2, curvas_etiquetas=True, for_pdf=False):
    x, y, z = data["X"].to_numpy(), data["Y"].to_numpy(), data["Z"].to_numpy()
    triang = mtri.Triangulation(x, y)
    fig, ax = plt.subplots(figsize=(8.3, 7.3) if for_pdf else (10, 10), dpi=140)
    mdt = ax.tripcolor(triang, z, shading="gouraud", cmap=cmap)

    if mostrar_curvas:
        start = math.floor(float(np.nanmin(z)) / intervalo) * intervalo
        end = math.ceil(float(np.nanmax(z)) / intervalo) * intervalo
        levels = np.arange(start, end + intervalo, intervalo)
        if len(levels) > 1:
            cs = ax.tricontour(triang, z, levels=levels, linewidths=0.8)
            if curvas_etiquetas:
                ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    ax.scatter(x, y, s=12)
    ax.set_title("Modelo Digital del Terreno", fontsize=13, fontweight="bold")
    ax.set_xlabel("X / Este")
    ax.set_ylabel("Y / Norte")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    fig.colorbar(mdt, ax=ax, label="Cota (m)")
    fig.tight_layout()
    return fig

def create_profile_figure(perfil, title_extra="", for_pdf=False):
    fig, ax = plt.subplots(figsize=(11.69, 5.2) if for_pdf else (15, 6), dpi=140)
    perfil = calcular_distancia_acumulada(perfil)
    x = perfil["Distancia acumulada"].to_numpy()
    y = perfil["Z"].to_numpy()
    ymin = math.floor(float(np.nanmin(y))*2)/2
    ymax = math.ceil(float(np.nanmax(y))*2)/2

    ax.set_facecolor("#FAFAFA")
    ax.fill_between(x, y, ymin, alpha=0.14)
    ax.plot(x, y, linewidth=2.4, marker="o", markersize=4.5, label="Perfil relevado")
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_formatter(FuncFormatter(pk_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="major", linewidth=0.8, alpha=0.55)
    ax.grid(which="minor", linewidth=0.35, alpha=0.25)
    ax.set_title(f"Perfil longitudinal {title_extra}\nLongitud: {x[-1]:.2f} m", fontsize=13, fontweight="bold")
    ax.set_xlabel("Progresiva / Distancia acumulada (m)")
    ax.set_ylabel("Cota (m)")

    idx_min = int(np.nanargmin(y))
    idx_max = int(np.nanargmax(y))
    ax.scatter([x[idx_min]], [y[idx_min]], s=60, zorder=5)
    ax.scatter([x[idx_max]], [y[idx_max]], s=60, zorder=5)
    ax.annotate(f"Mín: {y[idx_min]:.3f} m\nPK {format_pk(x[idx_min])}", xy=(x[idx_min], y[idx_min]), xytext=(15, -35), textcoords="offset points", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.65"), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate(f"Máx: {y[idx_max]:.3f} m\nPK {format_pk(x[idx_max])}", xy=(x[idx_max], y[idx_max]), xytext=(-80, 25), textcoords="offset points", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.65"), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.legend()
    fig.tight_layout()
    return fig, perfil

def plot_talud(H, V, punto_sup, punto_inf, relacion, for_pdf=False):
    fig, ax = plt.subplots(figsize=(7.8, 4.2) if for_pdf else (7.2, 4.3), dpi=140)
    ax.set_facecolor("#FAFAFA")
    ax.plot([0, H], [V, 0], linewidth=2.5, marker="o")
    ax.plot([0, 0], [0, V], linestyle="--", linewidth=1.2)
    ax.plot([0, H], [0, 0], linestyle="--", linewidth=1.2)
    ax.text(0, V, f"  Superior\n  Pto {punto_sup}", va="bottom", fontsize=9)
    ax.text(H, 0, f"  Inferior\n  Pto {punto_inf}", va="top", fontsize=9)
    ax.text(H/2, -0.08*max(V,1), f"H = {H:.3f} m", ha="center", va="top")
    ax.text(-0.05*max(H,1), V/2, f"V = {V:.3f} m", ha="right", va="center", rotation=90)
    ax.set_title(f"Talud aguas abajo = 1V : {relacion:.2f}H", fontsize=13, fontweight="bold")
    ax.set_xlabel("Horizontal (m)")
    ax.set_ylabel("Vertical (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.15*max(H,1), H*1.15)
    ax.set_ylim(-0.20*max(V,1), V*1.25)
    fig.tight_layout()
    return fig

def add_header(ax, title):
    ax.axis("off")
    ax.text(0.03, 0.96, title, fontsize=18, fontweight="bold", transform=ax.transAxes, va="top")
    ax.plot([0.03, 0.97], [0.91, 0.91], transform=ax.transAxes, linewidth=1.2)

def text_lines_page(title, lines):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=140)
    add_header(ax, title)
    y = 0.86
    for line in lines:
        if y < 0.08:
            break
        ax.text(0.06, y, line, fontsize=10.5, transform=ax.transAxes, va="top")
        y -= 0.04 if line else 0.025
    fig.tight_layout()
    return fig

def table_page(title, df_table, max_rows=32):
    fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=140)
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=18)
    show = df_table.head(max_rows).copy()
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    fig.tight_layout()
    return fig

def make_planilla_lines(form, auto):
    return [
        "PLANILLA INSPECCIÓN",
        "",
        "1. DATOS GENERALES",
        f"1.1 N° de represa: {form['n_represa']}",
        f"1.2 Fecha inspección: {form['fecha']}",
        f"1.3 Curso embalsado: {form['curso']}",
        f"1.4 Nombre del propietario: {form['propietario']}",
        f"1.5 Padrones represa: {form['padrones']}",
        f"    Sección Judicial: {form['seccion']}",
        f"    Departamento: {form['departamento']}",
        "",
        "2. DATOS REPRESA",
        f"2.1 Ancho mínimo de coronamiento: {auto.get('ancho_min', 'S/D')}",
        f"2.2 Sección de máxima altura:",
        f"    Altura del terraplén: {auto.get('altura_max', 'S/D')}",
        f"    Ancho de coronamiento: {auto.get('ancho_prom', 'S/D')}",
        "",
        "2.3 TALUD AGUAS ARRIBA",
        f"    {form['talud_arriba']}",
        "",
        "2.4 TALUD AGUAS ABAJO",
        f"    {form['talud_abajo']} Pendiente calculada: {auto.get('talud', 'S/D')}",
        "",
        f"2.5 Cota del Pelo de agua: {form['pelo_agua']}",
        f"2.6 Cota del Coronamiento: {auto.get('cota_corona', 'S/D')}",
        f"2.7 Escala: {form['escala']}",
        f"    Tipo y ubicación aproximada: {form['escala_ubicacion']}",
        f"2.8 Mojón de referencia: {form['mojon']}",
        f"    Ubicación aproximada: {form['mojon_ubicacion']}",
        f"    Cota: {form['mojon_cota']}",
        f"2.9 Origen utilizado en la nivelación: {form['origen']}",
    ]

def make_planilla_lines_2(form, auto):
    return [
        "2.10 OBRAS DE TOMA",
        f"Ubicación: {form['toma_ubicacion']}",
        f"Tipo y diámetro: {form['toma_tipo']}",
        f"Cota de zampeado aguas abajo: {form['toma_zampeado']}",
        "",
        "2.11 SEGURIDAD DE LA REPRESA",
        f"Situación observada hace temer por la seguridad de la obra: {form['seguridad']}",
        f"Comentarios: {form['seguridad_comentarios']}",
        "",
        "2.12 VERTEDERO",
        f"Franquía: {auto.get('franquia', 'S/D')}",
        f"Ubicación: {form['vertedero_ubicacion']}",
        f"Cotas del vertedero: {auto.get('cotas_vertedero', 'S/D')}",
        f"Comentarios: {form['vertedero_comentarios']}",
        "",
        "2.13 OBSERVACIONES",
        form["observaciones"],
        "",
        "2.14 Forma de acceso a la represa",
        form["acceso"],
        "",
        f"2.15 Inspeccionada por: {form['inspectores']}",
    ]

def build_pdf(form, auto, data, fig_mdt, fig_profile, anchos_df, talud_info, fig_talud, perfil_table):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(text_lines_page("Planilla de inspección", make_planilla_lines(form, auto)), bbox_inches="tight")
        plt.close()
        pdf.savefig(text_lines_page("Planilla de inspección", make_planilla_lines_2(form, auto)), bbox_inches="tight")
        plt.close()

        pdf.savefig(fig_mdt, bbox_inches="tight")
        plt.close(fig_mdt)
        pdf.savefig(fig_profile, bbox_inches="tight")
        plt.close(fig_profile)

        if anchos_df is not None and len(anchos_df):
            resumen_anchos = pd.DataFrame({
                "Indicador": ["Anchos calculados", "Ancho mínimo", "Ancho máximo", "Ancho promedio"],
                "Valor": [
                    len(anchos_df),
                    f"{anchos_df['Ancho corona (m)'].min():.3f} m",
                    f"{anchos_df['Ancho corona (m)'].max():.3f} m",
                    f"{anchos_df['Ancho corona (m)'].mean():.3f} m",
                ]
            })
            pdf.savefig(table_page("Resumen de anchos de coronamiento", resumen_anchos), bbox_inches="tight")
            plt.close()
            pdf.savefig(table_page("Tabla de anchos de coronamiento", anchos_df.round(3)), bbox_inches="tight")
            plt.close()

        if talud_info:
            talud_df = pd.DataFrame({
                "Parámetro": ["Punto superior", "Punto inferior", "Distancia horizontal H", "Diferencia vertical V", "Relación"],
                "Valor": [talud_info["p_sup"], talud_info["p_inf"], f"{talud_info['H']:.3f} m", f"{talud_info['V']:.3f} m", f"1V : {talud_info['relacion']:.2f}H"]
            })
            pdf.savefig(table_page("Talud aguas abajo", talud_df), bbox_inches="tight")
            plt.close()
            pdf.savefig(fig_talud, bbox_inches="tight")
            plt.close(fig_talud)

        if perfil_table is not None and len(perfil_table):
            pdf.savefig(table_page("Tabla del perfil longitudinal", perfil_table.round(3)), bbox_inches="tight")
            plt.close()

    buffer.seek(0)
    return buffer

if uploaded:
    df = read_file(uploaded)
    st.subheader("Vista previa")
    st.dataframe(df.head(30), use_container_width=True)

    cols = list(df.columns)
    default_num = cols.index("Numero") if "Numero" in cols else 0
    default_x = cols.index("X") if "X" in cols else min(1, len(cols)-1)
    default_y = cols.index("Y") if "Y" in cols else min(2, len(cols)-1)
    default_z = cols.index("Z") if "Z" in cols else min(3, len(cols)-1)
    if "Descripción" in cols:
        default_desc = cols.index("Descripción")
    elif "Descripcion" in cols:
        default_desc = cols.index("Descripcion")
    else:
        default_desc = min(4, len(cols)-1)

    st.subheader("Seleccionar columnas")
    c1, c2, c3, c4, c5 = st.columns(5)
    num_col = c1.selectbox("Numero / ID", cols, index=default_num)
    x_col = c2.selectbox("X / Este", cols, index=default_x)
    y_col = c3.selectbox("Y / Norte", cols, index=default_y)
    z_col = c4.selectbox("Z / Cota", cols, index=default_z)
    desc_col = c5.selectbox("Descripción", cols, index=default_desc)

    data = df[[num_col, x_col, y_col, z_col, desc_col]].copy()
    data.columns = ["Numero", "X", "Y", "Z", "Descripcion"]
    data["X"] = pd.to_numeric(data["X"], errors="coerce")
    data["Y"] = pd.to_numeric(data["Y"], errors="coerce")
    data["Z"] = pd.to_numeric(data["Z"], errors="coerce")
    data = data.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)
    data["Descripcion_norm"] = data["Descripcion"].apply(norm_text)

    if len(data) < 3:
        st.error("Para generar MDT necesitás al menos 3 puntos.")
        st.stop()

    descripciones = sorted(data["Descripcion_norm"].dropna().unique().tolist())

    st.subheader("Resumen general")
    a, b, c, d = st.columns(4)
    a.metric("Puntos", len(data))
    b.metric("Cota mínima", f"{data['Z'].min():.3f} m")
    c.metric("Cota máxima", f"{data['Z'].max():.3f} m")
    d.metric("Desnivel", f"{data['Z'].max() - data['Z'].min():.3f} m")

    st.divider()
    st.header("Formulario de inspección")

    with st.expander("1. Datos generales", expanded=True):
        g1, g2 = st.columns(2)
        n_represa = g1.text_input("1.1 N° de represa", "")
        fecha = g2.date_input("1.2 Fecha inspección")
        curso = st.text_input("1.3 Curso embalsado", "")
        propietario = st.text_input("1.4 Nombre del propietario", "")
        p1, p2, p3 = st.columns(3)
        padrones = p1.text_input("1.5 Padrones represa", "")
        seccion = p2.text_input("Sección Judicial", "")
        departamento = p3.text_input("Departamento", "")

    with st.expander("2. Datos descriptivos de represa", expanded=True):
        talud_arriba = st.text_area("2.3 Talud aguas arriba", "Describir estado actual: pendiente, enrocado, erosiones.")
        talud_abajo_txt = st.text_area("2.4 Talud aguas abajo", "Describir estado actual: pendiente, filtraciones, empastado, erosiones.")
        pelo_agua = st.text_input("2.5 Cota del pelo de agua", "")
        escala = st.selectbox("2.7 Escala", ["NO", "SI"])
        escala_ubicacion = st.text_input("Tipo y ubicación aproximada de escala", "")
        mojon = st.selectbox("2.8 Mojón de referencia", ["SI", "NO"])
        mojon_ubicacion = st.text_input("Ubicación aproximada del mojón", "")
        mojon_cota = st.text_input("Cota del mojón", "")
        origen = st.text_input("2.9 Origen utilizado en la nivelación", "Mojón")

    with st.expander("2.10 a 2.15 Obras, seguridad y observaciones", expanded=True):
        toma_ubicacion = st.text_input("2.10 Ubicación obra de toma", "")
        toma_tipo = st.text_input("Tipo y diámetro", "")
        toma_zampeado = st.text_input("Cota de zampeado aguas abajo", "")
        seguridad = st.selectbox("2.11 Seguridad de la represa", ["NO", "SI", "ATENCIÓN"])
        seguridad_comentarios = st.text_area("Comentarios seguridad", "")
        vertedero_ubicacion = st.text_input("2.12 Ubicación vertedero", "")
        vertedero_comentarios = st.text_area("Comentarios vertedero", "")
        observaciones = st.text_area("2.13 Observaciones", "")
        acceso = st.text_area("2.14 Forma de acceso a la represa", "")
        inspectores = st.text_input("2.15 Inspeccionada por", "")

    form = {
        "n_represa": n_represa, "fecha": fecha.strftime("%d/%m/%Y"), "curso": curso, "propietario": propietario,
        "padrones": padrones, "seccion": seccion, "departamento": departamento,
        "talud_arriba": talud_arriba, "talud_abajo": talud_abajo_txt, "pelo_agua": pelo_agua,
        "escala": escala, "escala_ubicacion": escala_ubicacion, "mojon": mojon, "mojon_ubicacion": mojon_ubicacion,
        "mojon_cota": mojon_cota, "origen": origen, "toma_ubicacion": toma_ubicacion, "toma_tipo": toma_tipo,
        "toma_zampeado": toma_zampeado, "seguridad": seguridad, "seguridad_comentarios": seguridad_comentarios,
        "vertedero_ubicacion": vertedero_ubicacion, "vertedero_comentarios": vertedero_comentarios,
        "observaciones": observaciones, "acceso": acceso, "inspectores": inspectores
    }

    st.divider()
    st.header("Cálculos automáticos")

    fc1, fc2, fc3 = st.columns(3)
    corona_sel = fc1.selectbox("Descripción de corona", descripciones, index=descripciones.index("corona") if "corona" in descripciones else 0)
    vertedero_sel = fc2.selectbox("Descripción de vertedero", descripciones, index=descripciones.index("vertedero") if "vertedero" in descripciones else (1 if len(descripciones)>1 else 0))
    base_sel = fc3.selectbox("Descripción de base", descripciones, index=descripciones.index("base") if "base" in descripciones else 0)

    corona = data[data["Descripcion_norm"] == corona_sel].copy()
    vertedero = data[data["Descripcion_norm"] == vertedero_sel].copy()
    base_points = data[data["Descripcion_norm"] == base_sel].copy()
    anchos_df = calcular_anchos_pares(base_points) if len(base_points) >= 2 else pd.DataFrame()

    auto = {}
    if len(corona):
        auto["cota_corona"] = f"Entre {corona['Z'].min():.3f} m y {corona['Z'].max():.3f} m"
        corona_prof = calcular_distancia_acumulada(corona.sort_index())
        auto["longitud"] = f"{corona_prof['Distancia acumulada'].iloc[-1]:.2f} m"
    else:
        auto["cota_corona"] = "S/D"
        auto["longitud"] = "S/D"

    if len(vertedero):
        auto["cotas_vertedero"] = f"Entre {vertedero['Z'].min():.3f} m y {vertedero['Z'].max():.3f} m"
    else:
        auto["cotas_vertedero"] = "S/D"

    if len(corona) and len(vertedero):
        franquia = float(corona["Z"].min() - vertedero["Z"].min())
        auto["franquia"] = f"{franquia:.3f} m"
    else:
        auto["franquia"] = "S/D"

    if len(anchos_df):
        auto["ancho_min"] = f"{anchos_df['Ancho corona (m)'].min():.3f} m"
        auto["ancho_prom"] = f"{anchos_df['Ancho corona (m)'].mean():.3f} m"
        auto["ancho_max"] = f"{anchos_df['Ancho corona (m)'].max():.3f} m"
    else:
        auto["ancho_min"] = auto["ancho_prom"] = auto["ancho_max"] = "S/D"

    auto["altura_max"] = f"{data['Z'].max() - data['Z'].min():.3f} m"

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Longitud corona", auto["longitud"])
    r2.metric("Franquía", auto["franquia"])
    r3.metric("Ancho mínimo", auto["ancho_min"])
    r4.metric("Altura / desnivel máx.", auto["altura_max"])

    if len(anchos_df):
        st.subheader("Tabla de anchos")
        st.dataframe(anchos_df.round(3), use_container_width=True)

    st.divider()
    st.header("MDT")
    m1, m2, m3, m4 = st.columns(4)
    cmap = m1.selectbox("Rampa de colores", ["terrain", "viridis", "plasma", "turbo", "jet", "gist_earth"])
    mostrar_curvas = m2.checkbox("Mostrar curvas", True)
    intervalo = m3.number_input("Intervalo curvas (m)", value=0.20, step=0.05, min_value=0.01)
    curvas_etiquetas = m4.checkbox("Etiquetar curvas", True)
    fig_mdt = create_mdt_figure(data, cmap, mostrar_curvas, intervalo, curvas_etiquetas)
    st.pyplot(fig_mdt)

    st.divider()
    st.header("Talud aguas abajo manual")
    opciones_puntos = [f"{row['Numero']} | {row['Descripcion']} | Z={row['Z']:.3f}" for _, row in data.iterrows()]
    t1, t2 = st.columns(2)
    punto_sup_txt = t1.selectbox("Punto superior", opciones_puntos, index=0)
    punto_inf_txt = t2.selectbox("Punto inferior", opciones_puntos, index=min(1, len(opciones_puntos)-1))
    ps = data.iloc[opciones_puntos.index(punto_sup_txt)]
    pi = data.iloc[opciones_puntos.index(punto_inf_txt)]
    H = math.sqrt((pi["X"] - ps["X"])**2 + (pi["Y"] - ps["Y"])**2)
    V = abs(pi["Z"] - ps["Z"])
    talud_info = None
    fig_talud = None
    if V == 0:
        st.error("La diferencia vertical es 0. No se puede calcular el talud.")
        auto["talud"] = "S/D"
    else:
        relacion = H / V
        auto["talud"] = f"1V : {relacion:.2f}H"
        talud_info = {"p_sup": ps["Numero"], "p_inf": pi["Numero"], "H": H, "V": V, "relacion": relacion}
        q1, q2, q3 = st.columns(3)
        q1.metric("H", f"{H:.3f} m")
        q2.metric("V", f"{V:.3f} m")
        q3.metric("Talud", auto["talud"])
        fig_talud = plot_talud(H, V, ps["Numero"], pi["Numero"], relacion)
        st.pyplot(fig_talud)

    st.divider()
    st.header("Perfil longitudinal")
    perfil_sel = st.multiselect("Descripción/es para perfil", descripciones, default=[corona_sel])
    perfil = data[data["Descripcion_norm"].isin(perfil_sel)].copy().sort_index().reset_index(drop=True)
    fig_profile = None
    perfil_table = None
    if len(perfil) >= 2:
        fig_profile, perfil_proc = create_profile_figure(perfil, ", ".join(perfil_sel))
        perfil_table = calcular_distancia_acumulada(perfil)[["Numero", "Descripcion", "Progresiva", "Distancia acumulada", "Z"]].rename(columns={"Z": "Cota"})
        st.pyplot(fig_profile)
        st.dataframe(perfil_table.round(3), use_container_width=True)
    else:
        st.info("Seleccioná al menos 2 puntos para el perfil.")

    st.divider()
    st.header("Generar PDF estilo planilla de inspección")
    if st.button("📄 Generar informe PDF"):
        if fig_profile is None:
            st.error("Primero generá un perfil válido.")
        elif fig_talud is None:
            st.error("Primero calculá un talud válido.")
        else:
            pdf_buffer = build_pdf(
                form, auto, data,
                create_mdt_figure(data, cmap, mostrar_curvas, intervalo, curvas_etiquetas, for_pdf=True),
                create_profile_figure(perfil, ", ".join(perfil_sel), for_pdf=True)[0],
                anchos_df,
                talud_info,
                plot_talud(talud_info["H"], talud_info["V"], talud_info["p_sup"], talud_info["p_inf"], talud_info["relacion"], for_pdf=True),
                perfil_table
            )
            st.download_button("Descargar PDF", pdf_buffer, "Planilla_Inspeccion_Represa.pdf", "application/pdf")
else:
    st.info("Subí tu archivo para empezar.")