import plotly.express as px
import plotly
import numpy as np


def plotly_visualizer(img, filename = "sample image"+ '.html'):
    """

    :param img: of size [nb_channels, height, width]
    :return:
    """
    img_to_visualize = np.transpose(img, (1, 2, 0))
    fig = px.imshow(
        img_to_visualize,
        color_continuous_scale="gray",
        aspect="equal",
    )
    fig.update_layout(coloraxis_showscale=False)
    plotly.offline.plot(fig, filename= filename,
                        auto_open=True)