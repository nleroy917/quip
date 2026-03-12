def show_image(img, title=None):
    """
    Open a PIL image in a matplotlib window.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_images(images, titles=None):
    """
    Display a list of PIL images side by side in a single matplotlib window.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()
