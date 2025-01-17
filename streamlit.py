import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.integrate import quad

# Sidebar Option Menu
with st.sidebar:
    selected = option_menu(
        "Metode Pilihan",
        [
            "Metode Newton Raphson", 
            "Metode Secant", 
             "Regresi Linear",
            "Interpolasi Lagrange",
            "Metode Pias", 
            "Metode Newton Cotes"
           
        ],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Metode Newton Raphson":
    st.title("Metode Newton Raphson")
    st.write("Masukkan fungsi f(x) dan turunannya df(x):")

    # Input fungsi
    fx_input = st.text_input("Masukkan f(x)", value="x**2 - 2*x - 2")
    dfx_input = st.text_input("Masukkan df(x)", value="2*x - 2")
    x0 = st.number_input("Masukkan nilai awal (x0)", value=2.0)
    tol = st.number_input("Toleransi error (e)", value=0.01)
    max_iter = st.number_input("Maksimum iterasi", value=100, step=1)

    if st.button("Hitung Newton Raphson"):
        f = lambda x: eval(fx_input)
        df = lambda x: eval(dfx_input)

        def newton_raphson(f, df, x0, tol=0.01, max_iter=100):
            iteration = 0
            while iteration < max_iter:
                x1 = x0 - f(x0) / df(x0)
                if abs(x1 - x0) < tol:
                    return x1, iteration
                x0 = x1
                iteration += 1
            return None, iteration

        root, iter_count = newton_raphson(f, df, x0, tol, max_iter)
        if root is not None:
            st.write(f"Akar ditemukan: {root:.6f} dalam {iter_count} iterasi.")
        else:
            st.write("Akar tidak ditemukan dalam batas iterasi.")

        # Plotting
        x_vals = np.linspace(x0 - 5, x0 + 5, 100)
        y_vals = [f(x) for x in x_vals]
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label="f(x)")
        plt.axhline(0, color="red", linestyle="--", label="y=0")
        if root:
            plt.scatter(root, f(root), color="blue", label=f"Root: {root:.6f}")
        plt.title("Newton Raphson Method")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        st.pyplot(plt)

elif selected == "Metode Secant":
    st.title("Metode Secant")
    st.write("Masukkan fungsi f(x):")

    fx_input = st.text_input("Masukkan f(x)", value="x**3 - 2*x**2 + 2*x - 5")
    x0 = st.number_input("Masukkan nilai awal pertama (x0)", value=2.0)
    x1 = st.number_input("Masukkan nilai awal kedua (x1)", value=3.0)
    tol = st.number_input("Toleransi error (e)", value=0.00001)
    max_iter = st.number_input("Maksimum iterasi", value=100, step=1)

    if st.button("Hitung Metode Secant"):
        f = lambda x: eval(fx_input)

        def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
            iteration = 0
            while iteration < max_iter:
                x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
                if abs(x2 - x1) < tol:
                    return x2, iteration
                x0, x1 = x1, x2
                iteration += 1
            return None, iteration

        root, iter_count = secant_method(f, x0, x1, tol, max_iter)
        if root is not None:
            st.write(f"Akar ditemukan: {root:.6f} dalam {iter_count} iterasi.")
        else:
            st.write("Akar tidak ditemukan dalam batas iterasi.")

        # Plotting
        x_vals = np.linspace(x0 - 1, x1 + 1, 100)
        y_vals = [f(x) for x in x_vals]
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label="f(x)")
        plt.axhline(0, color="red", linestyle="--", label="y=0")
        if root:
            plt.scatter(root, f(root), color="blue", label=f"Root: {root:.6f}")
        plt.title("Secant Method")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        st.pyplot(plt)

elif selected == "Interpolasi Lagrange":
    st.title("Interpolasi Lagrange")
    
    # Input data titik (x dan y)
    x_values = st.text_input("Masukkan x (dipisahkan dengan koma)", value="1, 2, 3")
    y_values = st.text_input("Masukkan y (dipisahkan dengan koma)", value="1, 4, 9")
    
    # Input untuk nilai x yang ingin dievaluasi
    eval_x = st.number_input("Masukkan nilai x untuk evaluasi", value=1.5)

    if st.button("Hitung Interpolasi Lagrange"):
        try:
            # Mengubah input menjadi list
            x = list(map(float, x_values.split(",")))
            y = list(map(float, y_values.split(",")))
            
            if len(x) != len(y):
                st.error("Jumlah elemen x dan y harus sama.")
            else:
                # Fungsi Lagrange Interpolation
                def lagrange_interpolation(x, y, eval_x):
                    def L(k, x_point):
                        return np.prod([(x_point - x[i]) / (x[k] - x[i]) for i in range(len(x)) if i != k])
                    return sum(y[k] * L(k, eval_x) for k in range(len(x)))

                result = lagrange_interpolation(x, y, eval_x)
                st.write(f"Interpolasi pada x={eval_x}: {result:.6f}")
                
                # Visualisasi
                x_plot = np.linspace(min(x) - 1, max(x) + 1, 500)
                y_plot = [lagrange_interpolation(x, y, xi) for xi in x_plot]
                plt.figure(figsize=(8, 6))
                plt.plot(x_plot, y_plot, label="Interpolasi Lagrange", color="blue")
                plt.scatter(x, y, color="red", label="Data Titik")
                plt.scatter(x, y, color="green", label="Data Interpolasi")

                # Tambahkan titik evaluasi pada plot
                plt.scatter(eval_x, result, color="green", zorder=5)
                plt.text(eval_x, result, f"({result:.6f})", color="green", fontsize=12, verticalalignment="bottom", horizontalalignment="right")
                
                plt.title("Interpolasi Lagrange")
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.legend()
                st.pyplot(plt)
        
        except Exception as e:
            st.error("Terjadi kesalahan dalam pemrosesan data.")
            st.error(e)



elif selected == "Metode Pias":
    st.title("Metode Pias")
    st.write("Hitung integral dengan kaidah Pias (Segiempat, Trapesium / Metode Trapezoidal, Titik Tengah):")

    # Input fungsi, batas, dan jumlah pias
    fx_input = st.text_input("Masukkan f(x)", value="x**2")
    a = st.number_input("Masukkan batas bawah (a)", value=0.0)
    b = st.number_input("Masukkan batas atas (b)", value=2.0)
    n = st.number_input("Masukkan jumlah pias (n)", value=10, step=1)

    # Pilih kaidah pias
    kaidah = st.selectbox(
        "Pilih kaidah pias:",
        options=["Segiempat", "Trapesium / Metode Trapezoidal", "Titik Tengah"],
        index=0,
    )

    if st.button("Hitung Metode Pias"):
        try:
            # Fungsi evaluasi
            f = lambda x: eval(fx_input)
            h = (b - a) / n  # Lebar tiap pias

            # Perhitungan berdasarkan kaidah
            if kaidah == "Segiempat":
                # Menggunakan nilai kiri
                x_points = np.linspace(a, b - h, n)  # Titik awal tiap pias
                result = h * sum(f(x) for x in x_points)

            elif kaidah == "Trapesium / Metode Trapezoidal":
                # Menggunakan rata-rata nilai di ujung pias
                x_points = np.linspace(a, b, n + 1)
                y_points = [f(x) for x in x_points]
                result = (h / 2) * (y_points[0] + 2 * sum(y_points[1:-1]) + y_points[-1])

            elif kaidah == "Titik Tengah":
                # Menggunakan titik tengah tiap pias
                x_points = np.linspace(a + h / 2, b - h / 2, n)  # Titik tengah tiap pias
                result = h * sum(f(x) for x in x_points)

            # Tampilkan hasil
            st.write(f"Hasil estimasi integral menggunakan kaidah {kaidah}: {result:.6f}")

            # Visualisasi
            x_plot = np.linspace(a, b, 500)
            y_plot = [f(x) for x in x_plot]
            plt.figure(figsize=(8, 6))
            plt.plot(x_plot, y_plot, label="f(x)", color="blue")

            if kaidah == "Segiempat":
                for x in x_points:
                    plt.bar(x, f(x), width=h, align="edge", color="lightgreen", edgecolor="black", alpha=0.7, label="Pias (Segiempat)" if x == x_points[0] else "")

            elif kaidah == "Trapesium / Metode Trapezoidal":
                for i in range(len(x_points) - 1):
                    plt.fill_between(
                        [x_points[i], x_points[i + 1]],
                        [f(x_points[i]), f(x_points[i + 1])],
                        color="lightcoral",
                        alpha=0.7,
                        label="Pias (Trapesium / Metode Trapezoidal)" if i == 0 else "",
                    )

            elif kaidah == "Titik Tengah":
                for x in x_points:
                    plt.bar(x, f(x), width=h, align="center", color="lightblue", edgecolor="black", alpha=0.7, label="Pias (Titik Tengah)" if x == x_points[0] else "")

            plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
            plt.title(f"Metode Pias - Kaidah {kaidah}")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error("Terjadi kesalahan dalam pemrosesan data.")
            st.error(e)

elif selected == "Metode Newton Cotes":
    st.title("Metode Newton Cotes")
    st.write("Hitung integral dengan kaidah Newton-Cotes (Trapesium / Metode Trapezoidal, Simpson 1/3 / Metode Parabolic, Simpson 3/8):")

    # Input fungsi, batas bawah, batas atas, dan jumlah subinterval
    fx_input = st.text_input("Masukkan f(x)", value="x**2")
    a = st.number_input("Masukkan batas bawah (a)", value=0.0)
    b = st.number_input("Masukkan batas atas (b)", value=2.0)
    n = st.number_input("Masukkan jumlah subinterval (n)", value=6, step=1)

    # Pilih kaidah integrasi
    kaidah = st.selectbox(
        "Pilih kaidah integrasi:",
        options=["Trapesium / Metode Trapezoidal", "Simpson 1/3 / Metode Parabolic", "Simpson 3/8"],
        index=0,
    )

    if st.button("Hitung Newton-Cotes"):
        try:
            # Fungsi evaluasi
            f = lambda x: eval(fx_input)
            h = (b - a) / n  # Panjang tiap subinterval

            if kaidah == "Trapesium / Metode Trapezoidal":
                # Kaidah Trapesium / Metode Trapezoidal
                x_points = np.linspace(a, b, n + 1)
                y_points = [f(x) for x in x_points]
                result = (h / 2) * (y_points[0] + 2 * sum(y_points[1:-1]) + y_points[-1])

            elif kaidah == "Simpson 1/3 / Metode Parabolic":
                # Kaidah Simpson 1/3 / Metode Parabolic
                if n % 2 != 0:
                    st.error("Untuk Simpson 1/3 / Metode Parabolic, jumlah subinterval (n) harus genap.")
                else:
                    x_points = np.linspace(a, b, n + 1)
                    y_points = [f(x) for x in x_points]
                    result = (h / 3) * (y_points[0] + 4 * sum(y_points[1:-1:2]) + 2 * sum(y_points[2:-2:2]) + y_points[-1])

            elif kaidah == "Simpson 3/8":
                # Kaidah Simpson 3/8
                if n % 3 != 0:
                    st.error("Untuk Simpson 3/8, jumlah subinterval (n) harus kelipatan 3.")
                else:
                    x_points = np.linspace(a, b, n + 1)
                    y_points = [f(x) for x in x_points]
                    result = (3 * h / 8) * (y_points[0] + 3 * sum(y_points[1:-1:3]) + 3 * sum(y_points[2:-1:3]) + 2 * sum(y_points[3:-1:3]) + y_points[-1])

            # Tampilkan hasil
            st.write(f"Hasil estimasi integral menggunakan kaidah {kaidah}: {result:.6f}")

            # Visualisasi
            x_plot = np.linspace(a, b, 500)
            y_plot = [f(x) for x in x_plot]
            plt.figure(figsize=(8, 6))
            plt.plot(x_plot, y_plot, label="f(x)", color="blue")

            if kaidah == "Trapesium / Metode Trapezoidal":
                for i in range(len(x_points) - 1):
                    plt.fill_between(
                        [x_points[i], x_points[i + 1]],
                        [f(x_points[i]), f(x_points[i + 1])],
                        color="lightgreen",
                        alpha=0.7,
                        label="Trapesium / Metode Trapezoidal" if i == 0 else "",
                    )

            elif kaidah == "Simpson 1/3 / Metode Parabolic" or kaidah == "Simpson 3/8":
                for i in range(len(x_points) - 1):
                    plt.fill_between(
                        [x_points[i], x_points[i + 1]],
                        [f(x_points[i]), f(x_points[i + 1])],
                        color="lightcoral",
                        alpha=0.7,
                        label=f"{kaidah}" if i == 0 else "",
                    )

            plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
            plt.title(f"Metode Newton-Cotes - Kaidah {kaidah}")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error("Terjadi kesalahan dalam pemrosesan data.")
            st.error(e)
            
elif selected == "Regresi Linear":
    st.title("Regresi Linear")
    # Input data hanya ditampilkan jika menu Regresi Linear dipilih
    st.write("Masukkan data untuk regresi linear:")
    col1, col2 = st.columns(2)
    with col1:
        x_input = st.text_area("Masukkan data (x), dipisahkan dengan koma:", value="1, 2, 3, 4, 5")
    with col2:
        y_input = st.text_area("Masukkan data (y), dipisahkan dengan koma:", value="5, 7, 7, 10, 16")

    # Input prediksi dinamis
    prediksi_RL = st.number_input("Masukkan jumlah X yang ingin prediksi:", min_value=0.0, value=6.0, step=0.1)

    if st.button("Hitung Regresi Linear"):
        try:
            # Parsing input
            x = np.array(list(map(float, x_input.split(",")))).reshape(-1, 1)
            y = np.array(list(map(float, y_input.split(","))))

            if len(x) != len(y):
                st.error("Jumlah elemen x dan y harus sama.")
            else:
                # Buat model regresi linear
                model = LinearRegression()
                model.fit(x, y)

                # Hasil regresi
                slope = model.coef_[0]
                intercept = model.intercept_
                r_squared = model.score(x, y)

                # Prediksi
                prediksi_hasil = model.predict([[prediksi_RL]])[0]

                # Menampilkan hasil
                st.write(f"### Hasil Analisis Regresi Linear:")
                st.write(f"1. **Korelasi (R²):** {r_squared:.4f}")
                st.write(f"2. **Persamaan regresi:** y = {intercept:.4f} + {slope:.4f}x")
                st.write(f"3. **Prediksi:** Jika jumlah X yang di prediksi {prediksi_RL:.1f} Maka jumlah Y adalah: {prediksi_hasil:.2f}")

                # Visualisasi data aktual dan garis regresi
                plt.figure(figsize=(8, 6))
                plt.scatter(x, y, color="blue", label="Data Aktual")
                plt.plot(x, model.predict(x), color="red", label="Garis Regresi")
                plt.axvline(x=prediksi_RL, color="green", linestyle="--", label="Jumlah X yang Prediksi")
                plt.scatter(prediksi_RL, prediksi_hasil, color="orange", label="Hasil Prediksi")
                plt.title("Regresi Linear")
                plt.xlabel("(x)")
                plt.ylabel("(y)")
                plt.legend()
                st.pyplot(plt)

        except Exception as e:
            st.error("Terjadi kesalahan dalam pemrosesan data.")
            st.error(e)
