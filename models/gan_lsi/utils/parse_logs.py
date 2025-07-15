#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv

def parse_log_file(log_path):
    """
    Lit un fichier de log et retourne une liste de tuples (epoch, C_real, C_fake, GP, LossC, LossG) sous forme de chaînes.
    """
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+—\s*C_real=([-\d.]+)\s+—\s*C_fake=([-\d.]+)\s+—\s*GP=([-\d.]+)\s+—\s*LossC=([-\d.]+)\s+—\s*LossG=([-\d.]+)"
    )
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append(m.groups())
    return rows

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir  = os.path.join(base_dir, "..", "logs")
    csv_dir  = os.path.join(base_dir, "..", "csv")

    os.makedirs(csv_dir, exist_ok=True)

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue

        log_path = os.path.join(log_dir, fname)
        csv_name = f"metrics_{os.path.splitext(fname)[0]}.csv"
        csv_path = os.path.join(csv_dir, csv_name)

        if os.path.exists(csv_path):
            print(f"Skipping {fname}: {csv_name} already exists")
            continue

        rows = parse_log_file(log_path)
        if not rows:
            print(f"Aucune métrique trouvée dans {fname}")
            continue

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(["epoch", "C_real", "C_fake", "GP", "LossC", "LossG"])
            for epoch, c_real, c_fake, gp, loss_c, loss_g in rows:
                # on remplace le séparateur décimal point par une virgule
                writer.writerow([
                    epoch,
                    c_real.replace('.', ','),
                    c_fake.replace('.', ','),
                    gp.replace('.', ','),
                    loss_c.replace('.', ','),
                    loss_g.replace('.', ','),
                ])

        print(f"Wrote {csv_path}")

if __name__ == "__main__":
    main()
