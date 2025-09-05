import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ---------- Input handling ----------
    if len(sys.argv) < 2:
        print("Usage: python image_processing_assignment.py <image.jpg>")
        return
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    # Load and resize to 512x512
    A_img = Image.open(img_path).convert("RGB")
    if A_img.size != (512, 512):
        A_img = A_img.resize((512, 512), Image.BILINEAR)
    A_img.save("A_input_512.jpg", quality=95, subsampling=0)
    A = np.array(A_img, dtype=np.uint8)

    h, w, _ = A.shape

    # ---------- 1) Display A ----------
    plt.imshow(A)
    plt.title("Image A (RGB)")
    plt.axis("off")
    plt.show()

    # # ---------- 2) Channel isolations ----------
    # RC = np.zeros((h, w), dtype=np.uint8)
    # GC = np.zeros((h, w), dtype=np.uint8)
    # BC = np.zeros((h, w), dtype=np.uint8)
    # for i in range(h):
    #     for j in range(w):
    #         r, g, b = A[i, j]
    #         RC[i, j] = r
    #         GC[i, j] = g
    #         BC[i, j] = b

    # Image.fromarray(RC).save("RC.jpg")
    # Image.fromarray(GC).save("GC.jpg")
    # Image.fromarray(BC).save("BC.jpg")

    # # Display
    # for channel, title in [(RC, "RC"), (GC, "GC"), (BC, "BC")]:
    #     plt.imshow(channel, cmap="gray", vmin=0, vmax=255)
    #     plt.title(title)
    #     plt.axis("off")
    #     plt.show()

        # ---------- 2) Channel isolations (colored only) ----------

    # Extract individual channels
    RC = A[:, :, 0]
    GC = A[:, :, 1]
    BC = A[:, :, 2]

    zeros = np.zeros((h, w), dtype=np.uint8)

    # Build colored versions
    R_img = np.stack([RC, zeros, zeros], axis=-1)  # Red channel image
    G_img = np.stack([zeros, GC, zeros], axis=-1)  # Green channel image
    B_img = np.stack([zeros, zeros, BC], axis=-1)  # Blue channel image

    # Save
    Image.fromarray(R_img).save("RC_color.jpg")
    Image.fromarray(G_img).save("GC_color.jpg")
    Image.fromarray(B_img).save("BC_color.jpg")

    # Display (no cmap!)
    for channel, title in [(R_img, "Red Channel Image"),
                           (G_img, "Green Channel Image"),
                           (B_img, "Blue Channel Image")]:
        plt.imshow(channel)
        plt.title(title)
        plt.axis("off")
        plt.show()


    # ---------- 3) Grayscale AG ----------
    AG = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            AG[i, j] = (int(RC[i, j]) + int(GC[i, j]) + int(BC[i, j])) // 3
    Image.fromarray(AG).save("AG.jpg")
    plt.imshow(AG, cmap="gray", vmin=0, vmax=255)
    plt.title("AG (Grayscale)")
    plt.axis("off")
    plt.show()

    # ---------- 4) Histograms ----------
    def histogram_counts(img2d):
        counts = [0] * 256
        hh, ww = img2d.shape
        for i in range(hh):
            for j in range(ww):
                counts[int(img2d[i, j])] += 1
        return counts

    for img2d, name in [(RC, "RC"), (GC, "GC"), (BC, "BC"), (AG, "AG")]:
        hist = histogram_counts(img2d)
        plt.bar(range(256), hist)
        plt.title(f"Histogram: {name}")
        plt.xlabel("Brightness value")
        plt.ylabel("Frequency")
        plt.show()

    # ---------- 5) Binarization ----------
    TB = int(input("Enter threshold TB for binarization (e.g. 100): "))
    AB = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            AB[i, j] = 255 if AG[i, j] >= TB else 0
    Image.fromarray(AB).save("AB.jpg")
    plt.imshow(AB, cmap="gray", vmin=0, vmax=255)
    plt.title(f"AB (Binary, TB={TB})")
    plt.axis("off")
    plt.show()

    # ---------- 6) Edge detection ----------
    TE = int(input("Enter threshold TE for edge detection (e.g. 15): "))
    Gx = np.zeros((h, w), dtype=np.int16)
    Gy = np.zeros((h, w), dtype=np.int16)
    for i in range(h):
        for j in range(w):
            if j < w - 1:
                Gx[i, j] = int(AG[i, j + 1]) - int(AG[i, j])
            if i < h - 1:
                Gy[i, j] = int(AG[i + 1, j]) - int(AG[i, j])

    GM = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            GM[i, j] = (Gx[i, j] ** 2 + Gy[i, j] ** 2) ** 0.5

    AE = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            AE[i, j] = 255 if GM[i, j] > TE else 0
    Image.fromarray(AE).save("AE.jpg")
    plt.imshow(AE, cmap="gray", vmin=0, vmax=255)
    plt.title(f"AE (Edges, TE={TE})")
    plt.axis("off")
    plt.show()

    # ---------- 7) Image Pyramid ----------
    def downsample_by_2(img2d):
        hh, ww = img2d.shape
        out = np.zeros((hh // 2, ww // 2), dtype=np.uint8)
        for i in range(hh // 2):
            for j in range(ww // 2):
                p1 = int(img2d[2 * i, 2 * j])
                p2 = int(img2d[2 * i, 2 * j + 1])
                p3 = int(img2d[2 * i + 1, 2 * j])
                p4 = int(img2d[2 * i + 1, 2 * j + 1])
                out[i, j] = (p1 + p2 + p3 + p4) // 4
        return out

    AG2 = downsample_by_2(AG)
    AG4 = downsample_by_2(AG2)
    AG8 = downsample_by_2(AG4)

    for img2d, name in [(AG2, "AG2"), (AG4, "AG4"), (AG8, "AG8")]:
        Image.fromarray(img2d).save(f"{name}.jpg")
        plt.imshow(img2d, cmap="gray", vmin=0, vmax=255)
        plt.title(name)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
