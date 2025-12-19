import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math


# ==========================================
# 1. BACKEND LOGIC (HELPER FUNCTIONS)
# ==========================================

def get_function_val(func_str, x_val):
    """
    Evaluasi fungsi string menjadi nilai float.
    """
    try:
        safe_dict = {
            "x": x_val,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "sqrt": np.sqrt,
            "log": np.log10,
            "ln": np.log,
            "pi": np.pi, "e": np.e,
            "abs": np.abs,
            "pow": np.power
        }
        return eval(func_str, {"__builtins__": None}, safe_dict)
    except Exception:
        return None


def get_symbolic_derivative(func_str, x_val):
    try:
        x = sp.symbols('x')
        clean_str = func_str.replace("e**", "E**").replace("e*", "E*").replace("(e)", "(E)")
        if clean_str == "e": clean_str = "E"
        clean_str = clean_str.replace("ln(", "NATURAL_LOG(")
        clean_str = clean_str.replace("log(", "log(10,")
        clean_str = clean_str.replace("NATURAL_LOG(", "log(")
        expr = sp.sympify(clean_str)
        derivative_expr = sp.diff(expr, x)
        val = derivative_expr.subs(x, x_val)
        exact_value = float(val.evalf())
        return exact_value, sp.latex(derivative_expr)
    except Exception as e:
        return None, str(e)


# --- Interpolasi Lagrange (Tetap) ---
def lagrange_interpolation(x, y, x_target):
    n = len(x)
    result = 0
    steps_data = []

    for i in range(n):
        row_data = {"i": i, "xi": x[i], "f(xi)": y[i]}
        L_i_val = 1.0
        coeff_counter = 1

        for j in range(n):
            if i != j:
                numerator = x_target - x[j]
                denominator = x[i] - x[j]
                factor = numerator / denominator
                row_data[f"Coeff {coeff_counter}"] = factor
                L_i_val *= factor
                coeff_counter += 1

        term = y[i] * L_i_val
        result += term
        steps_data.append(row_data)

    df = pd.DataFrame(steps_data)
    coeff_cols = [c for c in df.columns if c.startswith('Coeff')]
    coeff_cols.sort(key=lambda x: int(x.split(' ')[1]))
    final_cols = ['i', 'xi', 'f(xi)'] + coeff_cols
    df = df[final_cols]

    return result, df


# --- Interpolasi Newton (MODIFIED: Tabel Segitiga & Logic Error) ---
def newton_interpolation_detailed(x, y, x_target):
    n = len(x)

    # 1. Buat Matriks Divided Difference
    # Ukuran n x n (tapi kita hanya pakai separuh atas diagonal)
    dd_table = np.zeros((n, n))
    dd_table[:, 0] = y  # Kolom ke-0 adalah f(xi)

    # Hitung kolom selanjutnya
    for j in range(1, n):
        for i in range(n - j):
            # Rumus: (Next - Curr) / (x[i+j] - x[i])
            numerator = dd_table[i + 1, j - 1] - dd_table[i, j - 1]
            denominator = x[i + j] - x[i]
            dd_table[i, j] = numerator / denominator

    # Koefisien b0, b1, ... bn adalah baris pertama (dd_table[0, :])
    coefs = dd_table[0, :]

    # 2. Hitung Hasil Interpolasi (Polinomial)
    result = coefs[0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (x_target - x[i - 1])
        result += coefs[i] * product_term

    # 3. Format Output Tabel agar Segitiga (First, Second, etc)
    display_data = []
    col_names_map = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}

    for i in range(n):
        row = {"i": i, "xi": x[i], "f(xi)": y[i]}
        for j in range(1, n):
            col_name = col_names_map.get(j, f"Order {j}")

            # Logika Segitiga: Jika i < n - j, berarti ada isinya
            if i < n - j:
                row[col_name] = dd_table[i, j]
            else:
                row[col_name] = ""  # Kosongkan biar visualnya bagus
        display_data.append(row)

    df_display = pd.DataFrame(display_data)
    return result, df_display, coefs


def eval_newton_poly(coef, x_data, x_val):
    n = len(x_data)  # Hati-hati, len(coef) harus sama dengan len(x_data) yang dipakai membuat coef
    result = coef[0]
    for i in range(1, len(coef)):
        term = coef[i]
        for j in range(i):
            term *= (x_val - x_data[j])
        result += term
    return result


# ==========================================
# 2. STREAMLIT UI (FRONTEND)
# ==========================================

st.set_page_config(page_title="Project Metnum Kelompok", layout="wide")

# Sidebar
st.sidebar.title("Kalkulator Numerik")
menu = st.sidebar.selectbox("Pilih Metode:", ["Home", "Interpolasi", "Diferensiasi Numerik"])

if menu == "Home":
    st.title("Project Akhir Metode Numerik")
    st.markdown("""
    **Anggota Kelompok:**
    * Marco Alexander Tejo C14230262
    * Mario Vaun Goutama   C14230194

    **Update Fitur:**
    * **Newton:** Tabel Divided Difference format segitiga & Estimasi Error.
    """)

elif menu == "Interpolasi":
    st.title("Metode Interpolasi")
    st.caption("Metode: Lagrange [Cite: 8] & Newton (dengan Analisis Error) [Cite: 17]")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Input Data Utama")
        x_input = st.text_input("Titik X (koma):", "1, 4, 6")
        y_input = st.text_input("Titik Y (koma):", "0, 1.386294, 1.791759")
        x_target = st.number_input("Target X:", value=2.0)
        method = st.radio("Metode:", ["Lagrange", "Newton"])

        # FITUR BARU: Extra Point untuk Error Newton
        calc_error = False
        x_extra, y_extra = None, None

        if method == "Newton":
            st.markdown("---")
            st.subheader("2. Analisis Error (Opsional)")
            calc_error = st.checkbox("Hitung Estimasi Error (Butuh 1 titik tambahan)")
            if calc_error:
                st.info("Masukkan 1 titik data valid lainnya untuk menghitung taksiran error interpolasi.")
                c_ex1, c_ex2 = st.columns(2)
                x_extra = c_ex1.number_input("Titik X Tambahan:", value=5.0)
                y_extra = c_ex2.number_input("Titik Y Tambahan:", value=1.609438)

    try:
        # Parse Data Utama
        x_data = np.array([float(i) for i in x_input.split(',')])
        y_data = np.array([float(i) for i in y_input.split(',')])

        if len(x_data) != len(y_data):
            st.error("Jumlah data X dan Y harus sama!")
        else:
            df_input = pd.DataFrame({"X": x_data, "Y": y_data})
            with col2:
                st.write("Data Utama:")
                st.dataframe(df_input.T)

            if st.button("Hitung Interpolasi"):
                st.divider()

                # --- ALGORITMA LAGRANGE ---
                if method == "Lagrange":
                    res, df_steps = lagrange_interpolation(x_data, y_data, x_target)
                    st.success(f"Hasil Lagrange di x={x_target} adalah **{res:.6f}**")
                    st.write("Detail Faktor Lagrange:")
                    st.dataframe(df_steps.style.format("{:.6f}", subset=[c for c in df_steps.columns if c != 'i']))

                    # Plotting Lagrange
                    x_plot = np.linspace(min(x_data) - 1, max(x_data) + 1, 100)
                    y_plot = [lagrange_interpolation(x_data, y_data, xi)[0] for xi in x_plot]

                # --- ALGORITMA NEWTON ---
                else:
                    # 1. Hitung Newton Data Utama
                    res, df_steps, coefs = newton_interpolation_detailed(x_data, y_data, x_target)

                    st.success(f"Hasil Newton di x={x_target} adalah **{res:.6f}**")
                    st.write("Tabel Divided Difference:")
                    # Format tabel agar angka kosong tidak tampil sebagai NaN/0
                    st.dataframe(df_steps)

                    # 2. Hitung Error (Jika dicentang)
                    if calc_error:
                        # Gabungkan data utama + data extra
                        x_full = np.append(x_data, x_extra)
                        y_full = np.append(y_data, y_extra)

                        # Jalankan Newton pada data FULL untuk dapat koefisien selanjutnya
                        _, _, coefs_full = newton_interpolation_detailed(x_full, y_full, x_target)

                        # Koefisien terakhir dari data full (bn+1) adalah kunci error
                        b_next = coefs_full[-1]

                        # Rumus Error: b_next * (x - x0)(x - x1)...(x - xn)
                        # Kita hanya pakai product term dari data UTAMA
                        product_term = 1.0
                        for xi in x_data:
                            product_term *= (x_target - xi)

                        estimated_error = b_next * product_term

                        st.warning(f"**Estimasi Error:** {estimated_error:.8f}")
                        st.caption(
                            f"Error dihitung menggunakan titik tambahan ({x_extra}, {y_extra}) sebagai orde ke-{len(x_data)}.")

                    # Plotting Newton (pakai data utama saja untuk kurvanya)
                    x_plot = np.linspace(min(x_data) - 1, max(x_data) + 1, 100)
                    y_plot = [eval_newton_poly(coefs, x_data, xi) for xi in x_plot]

                # --- VISUALISASI UMUM ---
                st.subheader("Grafik")
                fig, ax = plt.subplots()
                ax.plot(x_plot, y_plot, 'b-', label='Interpolasi')
                ax.plot(x_data, y_data, 'ro', label='Data Utama')
                ax.scatter([x_target], [res], color='green', marker='x', s=100, zorder=5, label='Target')

                if method == "Newton" and calc_error:
                    ax.plot([x_extra], [y_extra], 'mo', label='Titik Bantu Error')

                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

    except ValueError:
        st.error("Input harus berupa angka.")

# --- HALAMAN DIFERENSIASI (Tetap Sama) ---
elif menu == "Diferensiasi Numerik":
    st.title("🔢 Diferensiasi Numerik: High Accuracy + Analisis")
    st.markdown("Membandingkan metode **Standard** vs **High Accuracy** serta validasi nilai eksak.")

    # --- 1. INPUT USER ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Konfigurasi")
        func_str = st.text_input("Masukkan Fungsi f(x):", "ln(x) + x**2",
                                 help="Support: ln(x), sin(x), exp(x), dll.")
        x_val = st.number_input("Titik evaluasi (x):", value=2.0, step=0.1)
        h_val = st.number_input("Step size (h):", value=0.1, format="%.4f", step=0.01)

    with col2:
        st.info("""
            **Tips Input:**
            - Logaritma natural: ketik `ln(x)`
            - Pangkat: ketik `x**2` (Python syntax)
            - Akar: `sqrt(x)`
            """)

    if st.button("Hitung Turunan"):
        # A. Hitung Nilai Eksak (True Value) pakai SymPy
        exact_val, deriv_latex = get_symbolic_derivative(func_str, x_val)

        if exact_val is None:
            st.error(f"Error parsing fungsi: {deriv_latex}")
        else:
            # Tampilkan Rumus Turunan Eksak
            st.markdown("---")
            st.subheader("1. Hasil Analisis Eksak (SymPy)")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Fungsi Asli:** $f(x) = {func_str}$")
            c1.markdown(f"**Turunan Simbolik:** $f'(x) = {deriv_latex}$")
            c2.metric("Nilai Eksak (True Value)", f"{exact_val:.8f}")

            # B. Hitung Numerik (Standard & High Acc)
            # Evaluasi titik stencil (x-2h s/d x+2h)
            f_x = get_function_val(func_str, x_val)
            f_xh = get_function_val(func_str, x_val + h_val)
            f_x2h = get_function_val(func_str, x_val + 2 * h_val)
            f_xmh = get_function_val(func_str, x_val - h_val)
            f_xm2h = get_function_val(func_str, x_val - 2 * h_val)

            if None in [f_x, f_xh, f_x2h, f_xmh, f_xm2h]:
                st.error("Error: Fungsi tidak valid atau domain error (misal ln negatif).")
            else:
                # Rumus Standard (2 Titik)
                std_fwd = (f_xh - f_x) / h_val
                std_bwd = (f_x - f_xmh) / h_val
                std_cen = (f_xh - f_xmh) / (2 * h_val)

                # Rumus High Accuracy (3-4 Titik)
                hi_fwd = (-3 * f_x + 4 * f_xh - f_x2h) / (2 * h_val)
                hi_bwd = (3 * f_x - 4 * f_xmh + f_xm2h) / (2 * h_val)
                hi_cen = (-f_x2h + 8 * f_xh - 8 * f_xmh + f_xm2h) / (12 * h_val)

                # C. Tabel Perbandingan Error
                st.subheader("2. Perbandingan Numerik vs Eksak")


                # Fungsi helper kecil untuk menghindari pembagian dengan nol
                def calc_error(true_val, calc_val):
                    if true_val == 0:
                        return 0.0  # Atau return abs(true_val - calc_val) jika ingin fallback ke absolute
                    return abs((true_val - calc_val) / true_val) * 100


                data_results = [
                    ["Forward (Standard)", std_fwd, calc_error(exact_val, std_fwd)],
                    ["Backward (Standard)", std_bwd, calc_error(exact_val, std_bwd)],
                    ["Central (Standard)", std_cen, calc_error(exact_val, std_cen)],
                    ["Forward (High Acc)", hi_fwd, calc_error(exact_val, hi_fwd)],
                    ["Backward (High Acc)", hi_bwd, calc_error(exact_val, hi_bwd)],
                    ["Central (High Acc)", hi_cen, calc_error(exact_val, hi_cen)],
                ]

                df_res = pd.DataFrame(data_results, columns=["Metode", "Hasil Hitung", "Error (%)"])

                # Menampilkan tabel dengan highlight error
                st.dataframe(df_res.style.format({
                    "Hasil Hitung": "{:.8f}",
                    "Error (%)": "{:.4f}%"  # Ubah format ini
                }).background_gradient(subset=["Error (%)"], cmap="Reds"), use_container_width=True)

                st.caption(f"*Error (%) = |(True - Numerik) / True| * 100. Semakin kecil semakin baik.")

                # D. Visualisasi
                st.subheader("3. Visualisasi & Analisis")

                tab1, tab2, tab3 = st.tabs(["Kurva & Garis Singgung", "Analisis Step Size (Log-Log)", "Rumus Manual"])

                # Tab 1: Visualisasi Konsep (Code 1 Style)
                with tab1:
                    x_plot = np.linspace(x_val - 3 * h_val, x_val + 3 * h_val, 100)
                    y_plot = [get_function_val(func_str, xi) for xi in x_plot]

                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.plot(x_plot, y_plot, label=f"f(x) = {func_str}", color='gray', alpha=0.5)

                    # Titik evaluasi
                    points_x = [x_val - 2 * h_val, x_val - h_val, x_val, x_val + h_val, x_val + 2 * h_val]
                    points_y = [f_xm2h, f_xmh, f_x, f_xh, f_x2h]
                    ax1.scatter(points_x, points_y, color='red', zorder=5, label='Titik Stencil')

                    # Garis Singgung (Pakai slope terbaik: Central High Acc)
                    tangent_line = hi_cen * (x_plot - x_val) + f_x
                    ax1.plot(x_plot, tangent_line, '--', color='blue', label=f"Garis Singgung (m={hi_cen:.4f})")

                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1)
                    st.caption("Garis putus-putus biru adalah taksiran garis singgung (turunan) pada titik x.")

                # Tab 2: Analisis Error vs h (Code 2 Style)
                with tab2:
                    st.markdown("Grafik ini menunjukkan seberapa cepat Error turun jika **h** diperkecil.")
                    h_range = np.logspace(-1, -5, 20)  # h dari 0.1 sampai 0.00001
                    err_fwd_std = []
                    err_cen_hi = []

                    for h_i in h_range:
                        # Evaluasi ulang titik untuk h berbeda
                        y_xh = get_function_val(func_str, x_val + h_i)
                        y_x = get_function_val(func_str, x_val)
                        y_xmh = get_function_val(func_str, x_val - h_i)
                        y_x2h = get_function_val(func_str, x_val + 2 * h_i)
                        y_xm2h = get_function_val(func_str, x_val - 2 * h_i)

                        # Hitung Error Forward Standard vs Central High Acc
                        val_fs = (y_xh - y_x) / h_i
                        val_ch = (-y_x2h + 8 * y_xh - 8 * y_xmh + y_xm2h) / (12 * h_i)

                        err_fwd_std.append(abs(exact_val - val_fs))
                        err_cen_hi.append(abs(exact_val - val_ch))

                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.loglog(h_range, err_fwd_std, 'o-', label='Forward Standard (Order 1)', color='red')
                    ax2.loglog(h_range, err_cen_hi, 's-', label='Central High Acc (Order 4)', color='blue')

                    ax2.set_xlabel('Step Size (h)')
                    ax2.set_ylabel('Absolute Error')
                    ax2.set_title('Log-Log Plot: Error vs Step Size')
                    ax2.grid(True, which="both", ls="--", alpha=0.4)
                    ax2.legend()
                    ax2.invert_xaxis()
                    st.pyplot(fig2)
                    st.info(
                        "Garis biru yang lebih curam menunjukkan metode High Accuracy jauh lebih cepat mencapai nilai eksak.")

                # Tab 3: Rumus (Code 1 Style)
                with tab3:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### Standard")
                        st.latex(r"f'(x_i) = \frac{f(x_{i+1}) - f(x_i)}{h}")
                        st.latex(r"f'(x_i) = \frac{f(x_i) - f(x_{i-1})}{h}")
                        st.latex(r"f'(x_i) = \frac{f(x_{i+1}) - f(x_{i-1})}{2h}")
                    with c2:
                        st.markdown("### High Accuracy")
                        st.latex(r"f'(x_i) = \frac{-f(x_{i+2}) + 4f(x_{i+1}) - 3f(x_i)}{2h}")
                        st.latex(r"f'(x_i) = \frac{3f(x_i) - 4f(x_{i-1}) + f(x_{i-2})}{2h}")
                        st.latex(r"f'(x_i) = \frac{-f(x_{i+2}) + 8f(x_{i+1}) - 8f(x_{i-1}) + f(x_{i-2})}{12h}")