# Fonte de informações de dados
## Atividade 1: Apresentação em equipe, utilizamos o software Canva:
### <img width="1035" height="587" alt="image" src="https://github.com/user-attachments/assets/26c22c02-6743-414e-a3ab-aa57ab5b6872" />

## Atividade Manipulação de Dados utilizando o Excel:
<img width="1756" height="695" alt="image" src="https://github.com/user-attachments/assets/aeaef94e-661b-409a-b1de-7124339e90d7" />

## Fórmulas utilizadas para o Desenvolvimento da tarefa:
CORRESP, ÍNDICE, CONT.SE, MÍNIMOSES, SOMASES. CONT.VALORES.

## Atividade de Introdução ao Power BI:
<img width="1305" height="785" alt="image" src="https://github.com/user-attachments/assets/e6f4917d-ee7f-4c68-9913-64aae5dd67b5" />

## Atividade sobre Regressão Linear:

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNKK7RbGGIWmtpnLV2jmXxv",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/GNeves92/Fonte-de-dados-/blob/main/analisesorvete1.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Análise de dados sobre o sorvete"
      ],
      "metadata": {
        "id": "hRKwaXWw2TcB"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LS_qS_DW2E5g",
        "outputId": "43c4a357-86e6-49cd-c9b0-4866288ec432"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading from https://www.kaggle.com/api/v1/datasets/download/sakshisatre/ice-cream-sales-dataset?dataset_version_number=1...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 2.09k/2.09k [00:00<00:00, 3.66MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Extracting files...\n",
            "caminho para os arquivos conjuntos de dados: /root/.cache/kagglehub/datasets/sakshisatre/ice-cream-sales-dataset/versions/1\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ],
      "source": [
        "import kagglehub\n",
        "\n",
        "# Download latest version\n",
        "path = kagglehub.dataset_download(\"sakshisatre/ice-cream-sales-dataset\")\n",
        "\n",
        "print(\"caminho para os arquivos conjuntos de dados:\", path)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "df_ice_cream = pd.read_csv(f\"{path}/Ice Cream.csv\")\n",
        "display(df_ice_cream.head(5))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "NC3Zk9w864q_",
        "outputId": "04e33f19-b6a1-461d-daeb-80d30104240b"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "   Temperature  Revenue\n",
              "0         24.6      535\n",
              "1         26.1      626\n",
              "2         27.8      661\n",
              "3         20.6      488\n",
              "4         11.6      317"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-132718db-1368-45fe-bfc6-d535110f2f43\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Temperature</th>\n",
              "      <th>Revenue</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>24.6</td>\n",
              "      <td>535</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>26.1</td>\n",
              "      <td>626</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>27.8</td>\n",
              "      <td>661</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>20.6</td>\n",
              "      <td>488</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>11.6</td>\n",
              "      <td>317</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-132718db-1368-45fe-bfc6-d535110f2f43')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-132718db-1368-45fe-bfc6-d535110f2f43 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-132718db-1368-45fe-bfc6-d535110f2f43');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-f7423bb9-26f2-4c11-a990-db52ab22b8c8\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-f7423bb9-26f2-4c11-a990-db52ab22b8c8')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-f7423bb9-26f2-4c11-a990-db52ab22b8c8 button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"display(df_ice_cream\",\n  \"rows\": 5,\n  \"fields\": [\n    {\n      \"column\": \"Temperature\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 6.46591060872326,\n        \"min\": 11.6,\n        \"max\": 27.8,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          26.1,\n          11.6,\n          27.8\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Revenue\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 135,\n        \"min\": 317,\n        \"max\": 661,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          626,\n          317,\n          661\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Create a scatter plot of Temperature vs Revenue\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.scatterplot(data=df_ice_cream, x='Temperature', y='Revenue')\n",
        "\n",
        "# Add titles and labels\n",
        "plt.title('Gráfico de Dispersão: Receita vs Temperatura do Sorvete')\n",
        "plt.xlabel('Temperatura (°C)')\n",
        "plt.ylabel('Venda')\n",
        "\n",
        "# Show the plot\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 565
        },
        "id": "47Xj7aDn894j",
        "outputId": "0b4cfb5a-4aeb-41ac-ef24-0e3b8ae87c46"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "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HtEw2QSo5DIc/TMf//z4Z9hsNihldqjkgtO47FYLHu1xK4pLTGh/S7i0/aVh7TDny+NIOXM9m9arlQZj7ozGs58dla7zS2YO7oqNcBqnyNBKg/AAmc9nic8c1SQ+b1TT+MzVfxX57GSCIAi+D6s9H330EV566SUcOnQIAQEBuPvuu9GlSxe8+eabWL9+PZKSkpwCIwC488470adPH7z66quYOHEizp07h++++07abzQaERQUhK+//hqDBg1yuefcuXMxb948l+3r16+HWq2u+hdJRERERET1gtFoxJgxY5Cfn4/Q0FCvx9bpzNb58+fx97//Hdu3b0dAQIDvE6rIzJkz8fTTT0tfFxQUIDo6Gv379/f5htYEi8WC7du3495774VC4X0RPNGN4vNWPc5kFWLYihSP+z+f1AsPvr0XALA8MQ5TNqR6PNZxv6GVBq891AmhgUpcyi/B7M3HsfdsaTZKrZBj2Zg4fLD7LPaWyZD9X/fm+PTwebRpGorjf+ZL1/CmoNiMf3521CnbJd7n2YFtEBcdDqPZimCVAppghc/rifjMUU3i80Y1jc9c/SfOeiuPOh1s/fTTT8jKysLtt1/vKWOz2bBr1y4sW7YM3333HcxmM/Ly8hAeHi4dc/nyZTRp0gQA0KRJExw8eNDpumK1QvGYslQqFVQqlct2hUJRp74p6tp4qGHj81a1im2AySbzuL/Icn2/WfDzeqy4v3esFi892Ama0EDkG814fvNJ7E67CqD03MfvaokVu35HSnqutA0AfkzLhU3ww4LhHXD491yM6a6DJtR3UQyNQoGFD3bBjI1HnUqyd2/ZCHe3aYKm4eUvrOEOnzmqSXzeqKbxmau/KvK51elg65577sGxY8ectiUlJaFNmzZ47rnnEB0dDYVCge+//x4jRowAAPz666/IzMxEz549AQA9e/bESy+9hKysLERFRQEAtm/fjtDQULRr165mXxARNXj5RjOyC80oKLEgNFABbZD7anmeSrOL/bNCAq7/ePZVdOLWiECsHNcN+shgKcApWy4e8N4va096DjKyjdjyywUYPBTscIcVAomIiDyr08FWSEgIOnTo4LQtKCgIGo1G2j5hwgQ8/fTTaNSoEUJDQzFt2jT07NkTPXr0AAD0798f7dq1wyOPPILFixfj0qVLePHFFzFlyhS32SsiosrIN5qRa7Rg1qZj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3Tpt2rRJffr0cbanpKTIbrertLTU5X7GHM6H2WxWenq6hg8frkWLFikrK0tPPvkk4w1tbteuXTp16pS+//3vKzw8XOHh4dq8ebOWLl2q8PBwJScnM+aCBGErQJjNZg0fPlwbNmxwtjU2NmrDhg3KycnpxJ4hGKSlpSklJcVl/JWXl+ujjz5i/OGcGIahWbNm6bXXXtPGjRuVlpbmcn348OEymUwuY+7AgQMqKipizKHNNDY2qra2lvGGNnf11Vdr37592rNnj/PXiBEj9LOf/cz5e8ZccGAZYQCZM2eOpkyZohEjRmjUqFFasmSJqqqqNG3atM7uGrqByspKFRQUOF8XFhZqz5496tGjh1JTU3XHHXfoj3/8ozIyMpSWlqb77rtPvXv31g033NB5nUbAmjlzplatWqXXX39dsbGxzj0K8fHxioqKUnx8vGbMmKE5c+aoR48eiouL0+zZs5WTk6NLL720k3uPQDRv3jxdd911Sk1NVUVFhVatWqX3339ff//73xlvaHOxsbHOPagO0dHRSkpKcrYz5oIDYSuA3HzzzTp9+rTmz5+vEydOaNiwYXrnnXfcihYA52Lnzp268sorna/nzJkjSZoyZYqef/55/f73v1dVVZV+9atfqbS0VGPGjNE777zDWnSck2eeeUaSdMUVV7i0r1y5UlOnTpUkPfHEEwoNDdWkSZNUW1urvLw8Pf300x3cU3QXp06d0i9+8QsdP35c8fHxyszM1N///nddc801khhv6HiMueAQYhiG0dmdAAAAAIDuhj1bAAAAANAOCFsAAAAA0A4IWwAAAADQDghbAAAAANAOCFsAAAAA0A4IWwAAAADQDghbAAAAANAOCFsAAASoAwcOKCUlRRUVFW32zLlz52r27Nlt9jwACGaELQDAeQkJCfH6a+HChZ3dxTbXv39/LVmypLO7oXnz5mn27NmKjY11tv3pT39Sv379lJ2drY8++sjlfsMw9N///d8aPXq0YmJilJCQoBEjRmjJkiWy2WySpLvuuksvvPCCDh8+3KE/CwB0R4QtAMB5OX78uPPXkiVLFBcX59J21113dXYX/WIYhurr6zv0M+12+zm/t6ioSOvWrdPUqVNd2h599FGtWbNGf/jDHzRt2jSX9/z85z/XHXfcoYkTJ2rTpk3as2eP7rvvPr3++ut69913JUlWq1V5eXl65plnzrlvAICzCFsAgPOSkpLi/BUfH6+QkBCXtjVr1mjQoEGKjIzUxRdfrKefftr53q+//lohISH661//qrFjxyoqKkojR47UV199pU8++UQjRoxQTEyMrrvuOp0+fdr5vqlTp+qGG27Q/fffr549eyouLk7//u//7hJeGhsbtWjRIqWlpSkqKkpZWVl6+eWXndfff/99hYSE6O2339bw4cMVERGhbdu26dChQ5o4caKSk5MVExOjkSNH6r333nO+74orrtCRI0d05513OmfvJGnhwoUaNmyYy3ezZMkS9e/f363fDz30kHr37q2BAwdKkv73f/9XI0aMUGxsrFJSUnTLLbfo1KlTXr/3v/71r8rKytKFF17obCsvL1dCQoIyMzM1fPhwVVdXu9z/4osvavXq1brnnns0cuRI9e/fXxMnTtTGjRt15ZVXOu+dMGGC1qxZ4/XzAQC+hXd2BwAA3deLL76o+fPn66mnnlJ2drby8/N16623Kjo6WlOmTHHet2DBAi1ZskSpqamaPn26brnlFsXGxurJJ5+UxWLRT37yE82fP99ltmXDhg2KjIzU+++/r6+//lrTpk1TUlKSHnroIUnSokWL9H//93969tlnlZGRoS1btujf/u3f1LNnT11++eXO58ydO1ePPfaYLrroIiUmJuro0aP6wQ9+oIceekgRERH6n//5H02YMEEHDhxQamqqXn31VWVlZelXv/qVbr311lZ/Jxs2bFBcXJzWr1/vbKurq9ODDz6ogQMH6tSpU5ozZ46mTp2qt956q8XnbN26VSNGjHBpGzJkiDIzMxUfHy+z2aw//elPLn8WAwcO1MSJE92eFRISovj4eOfrUaNG6ZtvvtHXX3/tEhYBAK1kAADQRlauXGnEx8c7Xw8YMMBYtWqVyz0PPvigkZOTYxiGYRQWFhqSjOeee855ffXq1YYkY8OGDc62RYsWGQMHDnS+njJlitGjRw+jqqrK2fbMM88YMTExRkNDg1FTU2NYLBbjgw8+cPnsGTNmGJMnTzYMwzA2bdpkSDLWrl3r8+e65JJLjGXLljlf9+vXz3jiiSdc7lmwYIGRlZXl0vbEE08Y/fr1c+l3cnKyUVtb6/XzPvnkE0OSUVFR0eI9WVlZxgMPPODxWnFxsWGz2VzaBg0aZFx//fVeP9ehrKzMkGS8//77ft0PAPCMmS0AQLuoqqrSoUOHNGPGDJcZoPr6epdZFEnKzMx0/j45OVmSNHToUJe25svqsrKyZLFYnK9zcnJUWVmpo0ePqrKyUjabTddcc43Le+x2u7Kzs13ams8OVVZWauHChXrzzTd1/Phx1dfXq7q6WkVFRa358Vs0dOhQmc1ml7Zdu3Zp4cKF+vTTT3XmzBk1NjZKOrsHa/DgwR6fU11drcjISI/XkpKS3NoMw/C7j1FRUZLkLJoBADg3hC0AQLuorKyUdLY63ujRo12uhYWFubw2mUzO3zv2QDVvcwSQ1nz2m2++6bKnSZIiIiJcXkdHR7u8vuuuu7R+/Xo99thjSk9PV1RUlH784x/7LGYRGhrqFmjq6urc7mv+eVVVVcrLy1NeXp5efPFF9ezZU0VFRcrLy/P6mVarVWfOnPHap6a+973v6csvv/Tr3m+//VaS1LNnT7+fDwBwR9gCALSL5ORk9e7dW4cPH9bPfvazNn/+p59+qurqaucszIcffqiYmBj17dtXPXr0UEREhIqKilz2Z/lj+/btmjp1qm688UZJZ4Pb119/7XKP2WxWQ0ODS1vPnj114sQJGYbhDIx79uzx+XlffvmlSkpKtHjxYvXt21eStHPnTp/vy87O1ueff+7HT3TWLbfcop/+9Kd6/fXX3fZtGYah8vJy54zj/v37ZTKZdMkll/j9fACAO6oRAgDazf33369FixZp6dKl+uqrr7Rv3z6tXLlSjz/++Hk/2263a8aMGfr888/11ltvacGCBZo1a5ZCQ0MVGxuru+66S3feeadeeOEFHTp0SLt379ayZcv0wgsveH1uRkaGXn31Ve3Zs0effvqpbrnlFrdZtf79+2vLli36xz/+oeLiYklnqxSePn1ajz76qA4dOqTly5fr7bff9vlzpKamymw2a9myZTp8+LDeeOMNPfjggz7fl5eXpx07driFvpb85Cc/0c0336zJkyfr4Ycf1s6dO3XkyBGtW7dO48eP16ZNm5z3bt261VkdEgBw7ghbAIB288tf/lLPPfecVq5cqaFDh+ryyy/X888/r7S0tPN+9tVXX62MjAyNGzdON998s66//nqXA5QffPBB3XfffVq0aJEGDRqkf/mXf9Gbb77p87Mff/xxJSYm6rLLLtOECROUl5en73//+y73PPDAA/r66681YMAA51K7QYMG6emnn9by5cuVlZWljz/+2K8zxnr27Knnn39eL730kgYPHqzFixfrscce8/m+6667TuHh4S5l6b0JCQnRqlWr9Pjjj2vt2rW6/PLLlZmZqYULF2rixInKy8tz3rtmzZpzqrQIAHAVYrRmxywAAF3A1KlTVVpaqrVr13Z2VzrV8uXL9cYbb+jvf/97mz3z7bff1u9+9zvt3btX4eHsNgCA88G/ogAABKhf//rXKi0tVUVFhWJjY9vkmVVVVVq5ciVBCwDaADNbAICAw8wWACAQELYAAAAAoB1QIAMAAAAA2gFhCwAAAADaAWELAAAAANoBYQsAAAAA2gFhCwAAAADaAWELAAAAANoBYQsAAAAA2gFhCwAAAADaAWELAAAAANrB/wfly90ksmgIawAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "import numpy as np\n",
        "\n",
        "# Prepare the data for the model\n",
        "# The independent variable (Temperature) needs to be in a 2D array/DataFrame\n",
        "X = df_ice_cream[['Temperature']]\n",
        "y = df_ice_cream['Revenue']\n",
        "\n",
        "# Create and train the linear regression model\n",
        "model = LinearRegression()\n",
        "model.fit(X, y)\n",
        "\n",
        "# Print the model's coefficients and intercept\n",
        "print(f\"Coeficiente (inclinação): {model.coef_[0]:.2f}\")\n",
        "print(f\"Intercepto: {model.intercept_:.2f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AXoahMqS8LOG",
        "outputId": "ce6c2910-bd24-4621-8e98-f67e48928328"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Coeficiente (inclinação): 21.44\n",
            "Intercepto: 44.30\n"
          ]
        }
      ]
    }
  ]
}




