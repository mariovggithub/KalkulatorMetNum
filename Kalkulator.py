import math


# FUNCTION TABEL
def print_table(headers, rows):
    col_widths = [len(str(h)) for h in headers]  # coba cobaa

    for row in rows:
        for i, item in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(item)))

    format_row = " | ".join("{:<" + str(w) + "}" for w in col_widths)
    print(format_row.format(*headers))
    print("-+-".join("-" * w for w in col_widths))

    for row in rows:
        print(format_row.format(*row))

# INTERPOLASI LAGRANGE
def lagrange_interpolation(x, y, x_target):
    n = len(x)
    result = 0
    table = []

    for i in range(n):
        term = y[i]
        steps = []
        for j in range(n):
            if i != j:
                term *= (x_target - x[j]) / (x[i] - x[j])
                steps.append(f"(x-{x[j]})/({x[i]}-{x[j]})")
        result += term
        table.append([f"L{i}", " * ".join(steps), round(term, 6)])

    return result, table

# INTERPOLASI NEWTON
def newton_interpolation(x, y, x_target):
    n = len(x)
    coef = y.copy()
    table = []

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])

    # tabel koefisien
    for i in range(n):
        table.append([i, round(coef[i], 6)])

    # evaluasi polinomial
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_target - x[i]) + coef[i]

    return result, table


# DIFERENSIASI NUMERIK
def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


# MENU
def main_menu():
    print("\n===== KALKULATOR NUMERIK =====")
    print("1. Interpolasi Lagrange")
    print("2. Interpolasi Newton")
    print("3. Diferensiasi Numerik")
    print("0. Keluar")
    return input("Pilih menu: ")


if __name__ == "__main__":
    while True:
        choice = main_menu()

        # ========= INTERPOLASI =========
        if choice in ["1", "2"]:
            n = int(input("Jumlah titik data: "))
            x = []
            y = []

            for i in range(n):
                xi = float(input(f"x[{i}]: "))
                yi = float(input(f"y[{i}]: "))
                x.append(xi)
                y.append(yi)

            x_target = float(input("Nilai x yang ingin dihitung: "))

            if choice == "1":
                result, table = lagrange_interpolation(x, y, x_target)
                print("\n--- HASIL INTERPOLASI LAGRANGE ---")
                print("Hasil =", result)

                print("\nTabel Perhitungan:")
                print_table(["Li", "Rumus", "Nilai"], table)

            else:
                result, table = newton_interpolation(x, y, x_target)
                print("\n--- HASIL INTERPOLASI NEWTON ---")
                print("Hasil =", result)

                print("\nTabel Koefisien Selisih Terbagi:")
                print_table(["i", "Koefisien"], table)

        # ========= DIFERENSIASI =========
        elif choice == "3":
            print("\nPilih fungsi:")
            print("1. f(x) = x^2")
            print("2. f(x) = sin(x)")
            f_choice = input("Pilihan: ")

            if f_choice == "1":
                f = lambda x: x**2
                df_true = lambda x: 2*x
            else:
                f = lambda x: math.sin(x)
                df_true = lambda x: math.cos(x)

            x = float(input("Nilai x: "))
            h = float(input("Nilai h: "))

            fwd = forward_difference(f, x, h)
            bwd = backward_difference(f, x, h)
            ctr = central_difference(f, x, h)
            true_val = df_true(x)

            table = [
                ["Forward", round(fwd, 6), round(abs(true_val - fwd), 6)],
                ["Backward", round(bwd, 6), round(abs(true_val - bwd), 6)],
                ["Central", round(ctr, 6), round(abs(true_val - ctr), 6)],
            ]

            print("\n--- HASIL DIFERENSIASI NUMERIK ---")
            print_table(["Metode", "Hasil", "Error"], table)

        # ========= KELUAR =========
        elif choice == "0":
            print("Program selesai.")
            break

        else:
            print("Pilihan tidak valid.")
