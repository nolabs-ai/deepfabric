(function () {
  let nextTableId = 0;

  function columns(value) {
    return new Set(
      (value || "")
        .split(",")
        .map((column) => Number(column.trim()) - 1)
        .filter((column) => Number.isInteger(column) && column >= 0),
    );
  }

  function initializeColumnToggles() {
    document.querySelectorAll(".column-toggle-table").forEach((container) => {
      if (container.dataset.columnTogglesReady) return;
      const table = container.querySelector("table");
      if (!table || !table.tHead) return;
      const toggleColumns = columns(container.dataset.toggleColumns);
      const hiddenColumns = columns(container.dataset.hiddenColumns);
      const headers = Array.from(table.tHead.rows[0].cells);
      const controls = document.createElement("div");
      controls.className = "table-column-controls";
      controls.setAttribute("role", "group");
      controls.setAttribute("aria-label", "Table column visibility");

      toggleColumns.forEach((index) => {
        if (index >= headers.length) return;
        if (!table.id) table.id = `column-toggle-table-${++nextTableId}`;
        const label = headers[index].textContent.trim();
        const button = document.createElement("button");
        button.type = "button";
        button.className = "table-column-toggle";
        button.setAttribute("aria-controls", table.id);
        let visible = !hiddenColumns.has(index);

        function render() {
          Array.from(table.rows).forEach((row) => {
            if (row.cells[index]) row.cells[index].hidden = !visible;
          });
          button.textContent = `${visible ? "Hide" : "Show"} ${label}`;
          button.setAttribute("aria-expanded", String(visible));
        }

        button.addEventListener("click", () => {
          visible = !visible;
          render();
        });
        render();
        controls.appendChild(button);
      });

      if (controls.childElementCount) {
        container.insertBefore(controls, container.firstChild);
        container.dataset.columnTogglesReady = "true";
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeColumnToggles);
  } else {
    initializeColumnToggles();
  }

  // Material's instant navigation replaces page content without reloading assets.
  if (typeof document$ !== "undefined") {
    document$.subscribe(initializeColumnToggles);
  }
})();
