(function () {
    const form = document.getElementById("analyse-form");
    const input = document.getElementById("file-input");
    const zone = document.getElementById("dropzone");
    const meta = document.getElementById("file-meta");
    const nameEl = document.getElementById("file-name");
    const sizeEl = document.getElementById("file-size");
    const errorEl = document.getElementById("file-error");
    const removeBtn = document.getElementById("file-remove");
    const submitBtn = document.getElementById("submit-btn");
    const MAX = 2 * 1024 * 1024;

    function showError(msg) {
        errorEl.hidden = !msg;
        errorEl.textContent = msg || "";
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    }

    function validate(file) {
        if (!file) return "Choose a CSV file.";
        const name = file.name.toLowerCase();
        if (!name.endsWith(".csv")) return "Only CSV files are accepted.";
        if (file.size > MAX) return "File must be 2 MB or smaller.";
        return "";
    }

    function syncFile() {
        const file = input.files && input.files[0];
        const err = file ? validate(file) : "";
        showError(err);
        if (file && !err) {
            meta.hidden = false;
            nameEl.textContent = file.name;
            sizeEl.textContent = "(" + formatSize(file.size) + ")";
        } else if (!file) {
            meta.hidden = true;
        }
        return !err && !!file;
    }

    zone.addEventListener("click", function (event) {
        if (event.target.closest("#file-remove")) return;
        input.click();
    });
    zone.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            input.click();
        }
    });
    input.addEventListener("change", syncFile);
    zone.addEventListener("dragover", function (event) {
        event.preventDefault();
        zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", function () {
        zone.classList.remove("dragover");
    });
    zone.addEventListener("drop", function (event) {
        event.preventDefault();
        zone.classList.remove("dragover");
        if (event.dataTransfer.files.length) {
            input.files = event.dataTransfer.files;
            syncFile();
        }
    });
    removeBtn.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        input.value = "";
        syncFile();
    });
    form.addEventListener("submit", function (event) {
        if (!syncFile()) {
            event.preventDefault();
            return;
        }
        submitBtn.disabled = true;
        submitBtn.querySelector(".btn-label").hidden = true;
        submitBtn.querySelector(".btn-loading").hidden = false;
    });
})();
