
def generate_display_art() -> str:
    letters = {
        'B': [
            " ###   ",
            "#   #  ",
            "#   #  ",
            "####   ",
            "#   #  ",
            "#   #  ",
            "####   "
        ],
        'U': [
            "#   #  ",
            "#   #  ",
            "#   #  ",
            "#   #  ",
            "#   #  ",
            "#   #  ",
            " ###   "
        ],
        'C': [
            " ####  ",
            "#    # ",
            "#      ",
            "#      ",
            "#      ",
            "#    # ",
            " ####  "
        ],
        'T': [
            "#######",
            "   #   ",
            "   #   ",
            "   #   ",
            "   #   ",
            "   #   ",
            "   #   "
        ],
        'O': [
            " ####  ",
            "#    # ",
            "#    # ",
            "#    # ",
            "#    # ",
            "#    # ",
            " ####  "
        ],
        'L': [
            "#      ",
            "#      ",
            "#      ",
            "#      ",
            "#      ",
            "#      ",
            "#######"
        ],
        'K': [
            "#   #  ",
            "#  #   ",
            "# #    ",
            "##     ",
            "# #    ",
            "#  #   ",
            "#   #  "
        ],
        'I': [
            "   #   ",
            "   #   ",
            "   #   ",
            "   #   ",
            "   #   ",
            "   #   ",
            "   #   "
        ]
    }

    # Target
    word = "BUCTOOLKIT"

    # 7 Lines to construct Letter Plot
    art_lines = ["" for _ in range(7)]
    for ch in word:
        ch_art = letters.get(ch, letters['O'])
        for i in range(7):
            art_lines[i] += ch_art[i] + " "     # one space as a delimiter

    # drop rest spaces
    art_lines = [line.rstrip() for line in art_lines]
    art_width = max(len(line) for line in art_lines)
    art_height = len(art_lines)

    # Monitor Shape
    left_margin = 4
    top_margin = 2
    internal_width = art_width + 2 * left_margin
    internal_height = art_height + 2 * top_margin

    # Inner Letters
    screen_lines = []
    for _ in range(internal_height):
        screen_lines.append([" "] * internal_width)

    # ASCII plot
    start_row = top_margin
    start_col = left_margin
    for i, line in enumerate(art_lines):
        for j, ch in enumerate(line):
            if start_row + i < internal_height and start_col + j < internal_width:
                screen_lines[start_row + i][start_col + j] = ch

    screen_content = ["".join(row) for row in screen_lines]

    # Monitor Edges
    top_border = "+" + "-" * internal_width + "+"
    bottom_border = "+" + "-" * internal_width + "+"
    bordered_lines = [top_border]
    for line in screen_content:
        bordered_lines.append("|" + line + "|")
    bordered_lines.append(bottom_border)

    # Monitor Steadier
    base_width = internal_width - 4
    if base_width < 6:
        base_width = 6
    base_pad = (internal_width - base_width) // 2
    base_line = " " * base_pad + "|" + "_" * (base_width - 2) + "|" + " " * (internal_width - base_pad - base_width)
    bordered_lines.append(base_line)
    foot_width = base_width + 2
    foot_pad = (internal_width - foot_width) // 2
    foot_line = " " * foot_pad + "|" + "_" * (foot_width - 2) + "|" + " " * (internal_width - foot_pad - foot_width)
    bordered_lines.append(foot_line)

    return "\n".join(bordered_lines)


if __name__ == "__main__":
    print(generate_display_art())