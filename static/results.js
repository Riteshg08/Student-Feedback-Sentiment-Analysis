(function () {
    const data = JSON.parse(document.getElementById("results-data").textContent);
    const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const colors = dark
        ? { positive: "#6fd196", neutral: "#e3b341", negative: "#ff8a80" }
        : { positive: "#1b7a3d", neutral: "#c48a12", negative: "#b42318" };
    const tick = dark ? "#f4efe6" : "#1c1916";

    function doughnut(id, cat) {
        const canvas = document.getElementById(id);
        if (!canvas || typeof Chart === "undefined") return;
        new Chart(canvas, {
            type: "doughnut",
            data: {
                labels: ["Positive", "Neutral", "Negative"],
                datasets: [
                    {
                        data: [cat.positive, cat.neutral, cat.negative],
                        backgroundColor: [colors.positive, colors.neutral, colors.negative],
                    },
                ],
            },
            options: {
                plugins: {
                    legend: { position: "bottom", labels: { color: tick } },
                },
            },
        });
    }

    data.categories.forEach(function (cat) {
        doughnut("chart-" + cat.id, cat);
    });

    const compare = document.getElementById("compare-chart");
    if (compare && typeof Chart !== "undefined") {
        new Chart(compare, {
            type: "bar",
            data: {
                labels: data.categories.map(function (c) { return c.name; }),
                datasets: [
                    {
                        label: "Positive",
                        data: data.categories.map(function (c) { return c.positive; }),
                        backgroundColor: colors.positive,
                    },
                    {
                        label: "Neutral",
                        data: data.categories.map(function (c) { return c.neutral; }),
                        backgroundColor: colors.neutral,
                    },
                    {
                        label: "Negative",
                        data: data.categories.map(function (c) { return c.negative; }),
                        backgroundColor: colors.negative,
                    },
                ],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                plugins: { legend: { position: "bottom", labels: { color: tick } } },
                scales: {
                    x: { stacked: true, ticks: { color: tick } },
                    y: { stacked: true, ticks: { color: tick } },
                },
            },
        });
    }

    const download = document.getElementById("download-csv");
    if (download) {
        download.addEventListener("click", function () {
            const rows = [["Category", "Positive", "Neutral", "Negative", "Blanks skipped", "Non-blank total"]];
            data.categories.forEach(function (c) {
                rows.push([c.name, c.positive, c.neutral, c.negative, c.blanks, c.total]);
            });
            const csv = rows
                .map(function (row) {
                    return row
                        .map(function (cell) {
                            const text = String(cell);
                            return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
                        })
                        .join(",");
                })
                .join("\n");
            const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "feedback-sentiment-results.csv";
            a.click();
            URL.revokeObjectURL(url);
        });
    }
})();
