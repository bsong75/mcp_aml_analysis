// Get filename from URL parameter
const urlParams = new URLSearchParams(window.location.search);
const filename = urlParams.get('file') || null;  // null = use most recent CSV

// Global variables
let ndx, all;
let dimensions = {};
let groups = {};
let charts = {};
let numericColumns = [];
let categoricalColumns = [];
let originalData = [];
let currentDashboard = 'dashboard1';

// Dashboard switching function
function switchDashboard(dashboardId) {
    currentDashboard = dashboardId;

    // Update tab active states
    document.querySelectorAll('.tab-button').forEach(tab => {
        tab.classList.remove('active');
    });
    const tabMap = {'dashboard1': '1', 'dashboard2': '2', 'dashboard3': '3'};
    document.getElementById(`tab-${tabMap[dashboardId]}`).classList.add('active');

    // Update dashboard title
    const titleElement = document.getElementById('dashboard-title');
    const titleMap = {
        'dashboard1': 'Interactive EDA Dashboard',
        'dashboard2': 'Dashboard 2',
        'dashboard3': 'Dashboard 3'
    };
    titleElement.textContent = titleMap[dashboardId];

    // Clear and regenerate dashboard content
    const container = document.getElementById('dashboard-content');
    container.innerHTML = '';

    if (dashboardId === 'dashboard1') {
        container.innerHTML = generateDashboardHTML();
        createCharts();
        updateAll();
    } else if (dashboardId === 'dashboard2') {
        container.innerHTML = generateDashboard2HTML();
        createDashboard2Charts();
    } else if (dashboardId === 'dashboard3') {
        container.innerHTML = generateDashboard3HTML();
        createDashboard3Charts();
    }
}

// Dashboard 2 - Sunburst Chart
function generateDashboard2HTML() {
    return `
        <div style="display: flex; gap: 15px;">
            <!-- Left: Sunburst Chart -->
            <div class="chart-container" style="flex: 1;">
                <div class="chart-title">
                    Hierarchical View: TP_FLAG → CTRY_CODE → CTRY_MONTH
                    <button class="reset-btn-small" onclick="resetAllFilters()">Reset</button>
                </div>
                <div id="sunburst-chart" style="width: 100%; height: 650px;"></div>
            </div>

            <!-- Right: Filtered Data Table -->
            <div class="chart-container" style="width: 450px; max-height: 730px; overflow-y: auto;">
                <div class="chart-title">📋 Filtered Data</div>
                <div id="plotly-table-dashboard2" style="width: 100%;"></div>
            </div>
        </div>
    `;
}

function createDashboard2Charts() {
    console.log('Creating Dashboard 2 charts, ndx:', ndx);
    createSunburstChart();
    updateDataTableDashboard2();
}

function createSunburstChart() {
    if (!ndx) {
        console.error('Crossfilter not initialized yet');
        return;
    }

    const allRecords = ndx.allFiltered();

    // Build hierarchical data structure
    const hierarchy = {
        name: 'Root',
        children: []
    };

    // Group by TP_FLAG -> CTRY_CODE -> CTRY_MONTH
    const tpFlagGroups = {};

    allRecords.forEach(d => {
        if (!d.TP_FLAG || !d.CTRY_CODE || !d.CTRY_MONTH) return;

        // TP_FLAG level
        if (!tpFlagGroups[d.TP_FLAG]) {
            tpFlagGroups[d.TP_FLAG] = {
                name: d.TP_FLAG,
                children: [],
                countries: {}
            };
        }

        // CTRY_CODE level
        if (!tpFlagGroups[d.TP_FLAG].countries[d.CTRY_CODE]) {
            tpFlagGroups[d.TP_FLAG].countries[d.CTRY_CODE] = {
                name: d.CTRY_CODE,
                children: [],
                months: {}
            };
        }

        // CTRY_MONTH level
        if (!tpFlagGroups[d.TP_FLAG].countries[d.CTRY_CODE].months[d.CTRY_MONTH]) {
            tpFlagGroups[d.TP_FLAG].countries[d.CTRY_CODE].months[d.CTRY_MONTH] = {
                name: d.CTRY_MONTH,
                value: 0
            };
        }

        tpFlagGroups[d.TP_FLAG].countries[d.CTRY_CODE].months[d.CTRY_MONTH].value++;
    });

    // Convert to Plotly sunburst format
    Object.values(tpFlagGroups).forEach(tpFlag => {
        Object.values(tpFlag.countries).forEach(country => {
            country.children = Object.values(country.months);
        });
        tpFlag.children = Object.values(tpFlag.countries).map(c => ({
            name: c.name,
            children: c.children
        }));
        hierarchy.children.push({
            name: tpFlag.name,
            children: tpFlag.children
        });
    });

    // Color mapping
    const colorMap = {
        'TP': '#F44336',
        'TN': '#4CAF50',
        'PN': '#FFEB3B'
    };

    // Flatten hierarchy for Plotly
    const labels = ['All'];
    const parents = [''];
    const values = [allRecords.length];
    const colors = ['#1a1a1a'];
    const text = ['All'];  // Display text (actual names)

    function addNode(node, parent, level = 0) {
        // Make labels unique by prepending parent for non-TP_FLAG levels
        let uniqueLabel = node.name;
        if (level === 1) {
            // CTRY_CODE level - make unique by prepending TP_FLAG
            uniqueLabel = `${parent}-${node.name}`;
        } else if (level === 2) {
            // CTRY_MONTH level - already unique with parent path
            uniqueLabel = `${parent}-${node.name}`;
        }

        labels.push(uniqueLabel);
        parents.push(parent);
        text.push(node.name);  // Display the actual name

        if (node.value !== undefined) {
            values.push(node.value);
        } else if (node.children) {
            // Sum of children
            const total = node.children.reduce((sum, child) => {
                return sum + (child.value || child.children?.reduce((s, c) => s + (c.value || 0), 0) || 0);
            }, 0);
            values.push(total);
        } else {
            values.push(0);
        }

        // Assign colors based on TP_FLAG
        if (colorMap[node.name]) {
            colors.push(colorMap[node.name]);
        } else if (parent === 'All' || colorMap[parent]) {
            // Child of TP_FLAG
            colors.push(colorMap[parent] || '#666666');
        } else {
            // CTRY_MONTH level - slightly darker
            colors.push('#555555');
        }

        if (node.children) {
            node.children.forEach(child => addNode(child, uniqueLabel, level + 1));
        }
    }

    hierarchy.children.forEach(child => addNode(child, 'All', 0));

    const data = [{
        type: 'sunburst',
        labels: labels,
        parents: parents,
        values: values,
        text: text,
        branchvalues: 'total',
        marker: {
            colors: colors,
            line: {
                color: '#ffffff',
                width: 2
            }
        },
        textfont: {
            color: '#ffffff',
            size: 11,
            family: 'Arial'
        },
        hovertemplate: '<b>%{text}</b><br>Count: %{value}<br>%{percentParent}<extra></extra>',
        insidetextorientation: 'radial'
    }];

    const layout = {
        margin: {t: 10, r: 10, b: 10, l: 10},
        paper_bgcolor: '#1a1a1a',
        plot_bgcolor: '#1a1a1a',
        height: 650
    };

    Plotly.newPlot('sunburst-chart', data, layout, {displayModeBar: false});

    // Add click handler for filtering
    const sunburstChart = document.getElementById('sunburst-chart');
    sunburstChart.on('plotly_sunburstclick', function(eventData) {
        const point = eventData.points[0];
        const actualValue = point.text;  // Use text instead of label (contains actual value)
        const label = point.label;  // Unique label for debugging

        console.log('Sunburst clicked:', actualValue, 'label:', label);

        // Check if "All" (center) is clicked - reset all filters
        if (actualValue === 'All' || label === 'All') {
            Object.values(dimensions).forEach(dim => dim.filterAll());
            updateAll();
            return;
        }

        // Determine which dimension to filter based on actual value
        if (actualValue === 'TP' || actualValue === 'TN' || actualValue === 'PN') {
            // TP_FLAG level
            if (dimensions.TP_FLAG.hasCurrentFilter()) {
                dimensions.TP_FLAG.filterAll();
            } else {
                dimensions.TP_FLAG.filter(actualValue);
            }
        } else if (actualValue.length === 2) {
            // Country code (2 letters like CA, FR, MX)
            if (!dimensions.CTRY_CODE) {
                dimensions.CTRY_CODE = ndx.dimension(d => d.CTRY_CODE || 'Unknown');
            }
            if (dimensions.CTRY_CODE.hasCurrentFilter()) {
                dimensions.CTRY_CODE.filterAll();
            } else {
                dimensions.CTRY_CODE.filter(actualValue);
            }
        } else if (actualValue.length === 4 || actualValue.includes('0') || actualValue.includes('1')) {
            // CTRY_MONTH (e.g., CA08, FR02)
            if (!dimensions.CTRY_MONTH) {
                dimensions.CTRY_MONTH = ndx.dimension(d => d.CTRY_MONTH || 'Unknown');
            }
            if (dimensions.CTRY_MONTH.hasCurrentFilter()) {
                dimensions.CTRY_MONTH.filterAll();
            } else {
                dimensions.CTRY_MONTH.filter(actualValue);
            }
        }

        // Update all charts (will refresh sunburst if on dashboard2)
        updateAll();
    });
}

function updateDataTableDashboard2() {
    const filteredData = Object.values(dimensions)[0]?.top(Infinity) || [];

    // Specify the columns we want to display
    const priorityColumns = ['TP_FLAG', 'MONTH', 'CTRY_CODE', 'CTRY_MONTH', 'ENTY_ID'];
    const displayColumns = priorityColumns.filter(col =>
        categoricalColumns.includes(col) || numericColumns.includes(col)
    );

    // Prepare table data for Plotly
    const headerValues = ['#', ...displayColumns];
    const cellValues = [
        // Row numbers
        filteredData.map((d, i) => i + 1),
        // Data columns
        ...displayColumns.map(col =>
            filteredData.map(d => {
                const value = d[col];
                return typeof value === 'number' ? value.toFixed(3) : value;
            })
        )
    ];

    const tableData = [{
        type: 'table',
        columnwidth: [40, ...displayColumns.map(() => 80)],
        header: {
            values: headerValues,
            align: 'center',
            line: {width: 1, color: '#374151'},
            fill: {color: '#242424'},
            font: {family: "Arial", size: 10, color: "white"}
        },
        cells: {
            values: cellValues,
            align: 'center',
            line: {width: 1, color: '#374151'},
            fill: {color: [
                filteredData.map((_, i) => i % 2 === 0 ? '#1a1a1a' : '#2a2a2a')
            ]},
            font: {family: "Arial", size: 9, color: "#a1a1aa"}
        }
    }];

    const layout = {
        margin: {t: 10, b: 10, l: 10, r: 10},
        paper_bgcolor: '#1a1a1a',
        plot_bgcolor: '#1a1a1a',
        height: 680
    };

    Plotly.newPlot('plotly-table-dashboard2', tableData, layout, {displayModeBar: false});
}

// Dashboard 3 - Placeholder for future use
function generateDashboard3HTML() {
    return `
        <div style="display: flex; justify-content: center; align-items: center; height: 650px;">
            <div style="text-align: center;">
                <h2 style="color: var(--text-primary); font-size: 2rem; margin-bottom: 20px;">Dashboard 3</h2>
                <p style="color: var(--text-secondary); font-size: 1.2rem;">Ready for your custom visualizations!</p>
                <p style="color: var(--text-muted); margin-top: 10px;">Add your charts here by editing generateDashboard3HTML() and createDashboard3Charts()</p>
            </div>
        </div>
    `;
}

function createDashboard3Charts() {
    // Placeholder for Dashboard 3 charts
    console.log('Dashboard 3 ready for implementation');
}

// Color schemes
const colorSchemes = {
    categorical: d3.schemeCategory10,
    sequential: d3.interpolateBlues,
    diverging: d3.interpolateRdBu
};

// Initialize dashboard
async function initDashboard() {
    try {
        // Fetch CSV data from backend
        // If filename is null, don't include it in query (backend will use most recent)
        const url = filename
            ? `/api/csv-data?filename=${encodeURIComponent(filename)}`
            : `/api/csv-data`;
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Update file info
        updateFileInfo(result.filename, result.data.length, Object.keys(result.data[0] || {}).length);
        
        // Process and load data
        processData(result.data);
        createDashboard();
        updateAll();
        
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('dashboard-content').innerHTML = `
            <div class="error">
                <h3>❌ Error Loading Data</h3>
                <p>${error.message}</p>
                <p>Please make sure the CSV file exists and is properly formatted.</p>
            </div>
        `;
    }
}

function updateFileInfo(name, rows, cols) {
    const nameElement = document.getElementById('dataset-name');
    if (nameElement) {
        nameElement.textContent = name;
    } else {
        console.error('dataset-name element not found in DOM');
    }
}

function processData(rawData) {
    if (!rawData || rawData.length === 0) {
        throw new Error('No data available');
    }
    
    // Convert data types and clean
    originalData = rawData.map((d, index) => {
        const processed = { _id: index };
        
        Object.keys(d).forEach(key => {
            let value = d[key];

            // Trim whitespace from string values
            if (typeof value === 'string') {
                value = value.trim();
            }

            // Skip ID columns for analysis
            if (key.toLowerCase().includes('id')) {
                processed[key] = value;
                return;
            }

            // Try to convert to number
            const numValue = +value;
            if (!isNaN(numValue) && isFinite(numValue)) {
                processed[key] = numValue;
                if (!numericColumns.includes(key)) {
                    numericColumns.push(key);
                }
            } else {
                processed[key] = String(value || '').trim();
                if (!categoricalColumns.includes(key)) {
                    categoricalColumns.push(key);
                }
            }
        });
        
        return processed;
    });
    
    console.log('Processed data:', {
        rows: originalData.length,
        numeric: numericColumns,
        categorical: categoricalColumns
    });
}

function createDashboard() {
    // Create crossfilter
    ndx = crossfilter(originalData);
    all = ndx.groupAll();
    
    // Create dimensions and groups
    numericColumns.forEach(col => {
        dimensions[col] = ndx.dimension(d => d[col]);
        groups[col] = dimensions[col].group();
    });
    
    categoricalColumns.forEach(col => {
        dimensions[col] = ndx.dimension(d => d[col]);
        groups[col] = dimensions[col].group();
    });
    
    // Generate dashboard HTML
    const dashboardHTML = generateDashboardHTML();
    document.getElementById('dashboard-content').innerHTML = dashboardHTML;
    
    // Create charts
    createCharts();
}

function generateDashboardHTML() {
    // Find the best columns for pie chart (categorical) and bar chart (numeric/time)
    const pieColumn = categoricalColumns[0] || numericColumns[0];
    const barColumn = numericColumns[0] || categoricalColumns[1] || categoricalColumns[0];

    let html = `
        <div style="display: flex; gap: 15px; margin-bottom: 15px;">
            <!-- Left side: Charts -->
            <div style="flex: 1; display: flex; flex-direction: column; gap: 15px;">
                <!-- First row of charts -->
                <div style="display: flex; gap: 15px;">
                    <div class="chart-container" style="width: 250px;">
                        <div class="chart-title" style="margin-bottom: 15px;">
                            Target by Category
                            <button class="reset-btn-small" onclick="resetTargetFilter()">Reset</button>
                        </div>
                        <div style="display: flex; justify-content: center; align-items: center;">
                            <div id="pie-chart" style="width: 210px; height: 190px;"></div>
                        </div>
                    </div>

                    <div class="chart-container" style="flex: 1;">
                        <div class="chart-title">
                            Monthly Imports
                            <button class="reset-btn-small" onclick="resetMonthFilter()">Reset</button>
                        </div>
                        <div id="bar-chart" style="width: 100%; height: 240px;"></div>
                    </div>

                    <div class="chart-container" style="width: 300px;">
                        <div class="chart-title">
                            Top 5 Countries
                            <button class="reset-btn-small" onclick="resetCountryFilter()">Reset</button>
                        </div>
                        <div id="country-chart" style="width: 100%; height: 240px;"></div>
                    </div>
                </div>

                <!-- Second row: Sankey chart -->
                <div class="chart-container" style="flex: 1;">
                    <div class="chart-title">
                        Sankey Chart
                        <button class="reset-btn-small" onclick="resetAllFilters()">Reset</button>
                    </div>
                    <div id="sankey-chart" style="width: 100%; height: 300px;"></div>
                </div>
            </div>

            <!-- Right side: Data table -->
            <div class="chart-container" style="width: 450px; max-height: 680px; overflow-y: auto;">
                <div class="chart-title">📋 Filtered Data</div>
                <div id="plotly-table" style="width: 100%;"></div>
            </div>
        </div>
    `;

    return html;
}

function createCharts() {
    const tooltip = d3.select('#tooltip');

    // Create pie chart with predefined categories
    charts.pie = createCategoryPieChart(tooltip);

    // Create monthly sales bar chart
    charts.bar = createMonthlySalesChart(tooltip);

    // Create top countries chart
    charts.country = createTopCountriesChart(tooltip);

    // Create Sankey chart
    charts.sankey = createSankeyChart();
}

function createCategoryPieChart(tooltip) {
    // Create TP_FLAG dimension if it doesn't exist
    if (!dimensions.TP_FLAG) {
        dimensions.TP_FLAG = ndx.dimension(d => d.TP_FLAG || 'Unknown');
        groups.TP_FLAG = dimensions.TP_FLAG.group().reduceCount();
    }

    // Color mapping for TP_FLAG categories
    const colorMap = {
        'TP': '#F44336',
        'TN': '#4CAF50',
        'PN': '#FFEB3B'
    };

    return {
        update: function() {
            // Get TP_FLAG counts from crossfilter
            const data = groups.TP_FLAG.all().filter(d => d.value > 0);
            console.log('TP_FLAG data from crossfilter:', data);

            if (data.length === 0) return;

            const labels = data.map(d => d.key);
            const values = data.map(d => d.value);
            const colors = data.map(d => colorMap[d.key] || '#999999');

            const pieData = [{
                labels: labels,
                values: values,
                type: 'pie',
                hole: 0.3,  // Donut chart
                marker: {
                    colors: colors,
                    line: {
                        color: '#ffffff',
                        width: 2
                    }
                },
                textinfo: 'label',
                textfont: {
                    color: '#ffffff',
                    size: 11,
                    family: 'Arial'
                },
                hovertemplate: '%{label}<br>Count: %{value}<extra></extra>'
            }];

            const layout = {
                width: 210,
                height: 210,
                margin: {t: 15, r: 5, b: 15, l: 5},
                paper_bgcolor: '#1a1a1a',
                plot_bgcolor: '#1a1a1a',
                showlegend: false
            };

            Plotly.newPlot('pie-chart', pieData, layout, {displayModeBar: false});

            // Add click handler for filtering
            const pieChart = document.getElementById('pie-chart');
            pieChart.removeAllListeners && pieChart.removeAllListeners('plotly_click');

            pieChart.on('plotly_click', function(data) {
                const category = data.points[0].label;
                if (dimensions.TP_FLAG.hasCurrentFilter()) {
                    dimensions.TP_FLAG.filterAll();
                } else {
                    dimensions.TP_FLAG.filter(category);
                }
                updateAll();
            });
        }
    };
}

function createMonthlySalesChart(tooltip) {
    // Create MONTH dimension if it doesn't exist
    if (!dimensions.MONTH) {
        dimensions.MONTH = ndx.dimension(d => d.MONTH || 'Unknown');
        groups.MONTH = dimensions.MONTH.group().reduceCount();
    }

    return {
        update: function() {
            const allRecords = ndx.allFiltered();
            const monthOrder = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

            // Count all TP_FLAG types per month
            const monthCounts = {};
            allRecords.forEach(record => {
                const month = record.MONTH;
                if (month) {
                    if (!monthCounts[month]) {
                        monthCounts[month] = {TP: 0, TN: 0, PN: 0};
                    }
                    const target = record.TP_FLAG;
                    if (target === 'TP' || target === 'TN' || target === 'PN') {
                        monthCounts[month][target]++;
                    }
                }
            });

            // Sort months and prepare data
            const months = Object.keys(monthCounts).sort((a, b) =>
                monthOrder.indexOf(a) - monthOrder.indexOf(b)
            );

            const tpCounts = months.map(m => monthCounts[m].TP);
            const tnCounts = months.map(m => monthCounts[m].TN);
            const pnCounts = months.map(m => monthCounts[m].PN);

            const traceTP = {
                x: months,
                y: tpCounts,
                name: 'TP',
                type: 'bar',
                marker: {color: '#F44336'},
                hovertemplate: '%{x}<br>TP: %{y}<extra></extra>'
            };

            const traceTN = {
                x: months,
                y: tnCounts,
                name: 'TN',
                type: 'bar',
                marker: {color: '#4CAF50'},
                hovertemplate: '%{x}<br>TN: %{y}<extra></extra>'
            };

            const tracePN = {
                x: months,
                y: pnCounts,
                name: 'PN',
                type: 'bar',
                marker: {color: '#FFEB3B'},
                hovertemplate: '%{x}<br>PN: %{y}<extra></extra>'
            };

            const layout = {
                height: 240,
                barmode: 'stack',
                margin: {t: 30, r: 20, b: 40, l: 40},
                paper_bgcolor: '#1a1a1a',
                plot_bgcolor: '#242424',
                xaxis: {
                    title: '',
                    tickfont: {color: '#a1a1aa', size: 10},
                    gridcolor: '#374151',
                    color: '#a1a1aa'
                },
                yaxis: {
                    title: 'Count',
                    tickfont: {color: '#a1a1aa', size: 10},
                    gridcolor: '#374151',
                    color: '#a1a1aa'
                },
                legend: {
                    orientation: 'h',
                    x: 0.5,
                    xanchor: 'center',
                    y: 1.15,
                    font: {color: '#a1a1aa', size: 9}
                }
            };

            Plotly.newPlot('bar-chart', [traceTP, traceTN, tracePN], layout, {displayModeBar: false});

            const barChart = document.getElementById('bar-chart');
            barChart.removeAllListeners && barChart.removeAllListeners('plotly_click');

            barChart.on('plotly_click', function(data) {
                const month = data.points[0].x;
                if (dimensions.MONTH.hasCurrentFilter()) {
                    dimensions.MONTH.filterAll();
                } else {
                    dimensions.MONTH.filter(month);
                }
                updateAll();
            });
        }
    };
}

function createTopCountriesChart(tooltip) {
    if (!dimensions.CTRY_CODE) {
        dimensions.CTRY_CODE = ndx.dimension(d => d.CTRY_CODE || 'Unknown');
        groups.CTRY_CODE = dimensions.CTRY_CODE.group().reduceCount();
    }

    return {
        update: function() {
            const allRecords = ndx.allFiltered();

            // Count all TP_FLAG types per country
            const countryCounts = {};
            allRecords.forEach(d => {
                if (d.CTRY_CODE && d.TP_FLAG) {
                    if (!countryCounts[d.CTRY_CODE]) {
                        countryCounts[d.CTRY_CODE] = {TP: 0, TN: 0, PN: 0, total: 0};
                    }
                    const target = d.TP_FLAG;
                    if (target === 'TP' || target === 'TN' || target === 'PN') {
                        countryCounts[d.CTRY_CODE][target]++;
                        countryCounts[d.CTRY_CODE].total++;
                    }
                }
            });

            // Sort by total count and get top 5
            const sortedCountries = Object.entries(countryCounts)
                .sort((a, b) => b[1].total - a[1].total)
                .slice(0, 5);

            const countries = sortedCountries.map(d => d[0]);
            const tpCounts = sortedCountries.map(d => d[1].TP);
            const tnCounts = sortedCountries.map(d => d[1].TN);
            const pnCounts = sortedCountries.map(d => d[1].PN);

            const traceTP = {
                x: tpCounts,
                y: countries,
                name: 'TP',
                type: 'bar',
                orientation: 'h',
                marker: {color: '#F44336'},
                hovertemplate: '%{y}<br>TP: %{x}<extra></extra>'
            };

            const traceTN = {
                x: tnCounts,
                y: countries,
                name: 'TN',
                type: 'bar',
                orientation: 'h',
                marker: {color: '#4CAF50'},
                hovertemplate: '%{y}<br>TN: %{x}<extra></extra>'
            };

            const tracePN = {
                x: pnCounts,
                y: countries,
                name: 'PN',
                type: 'bar',
                orientation: 'h',
                marker: {color: '#FFEB3B'},
                hovertemplate: '%{y}<br>PN: %{x}<extra></extra>'
            };

            const layout = {
                width: 280,
                height: 240,
                barmode: 'stack',
                margin: {t: 30, r: 10, b: 40, l: 30},
                paper_bgcolor: '#1a1a1a',
                plot_bgcolor: '#242424',
                xaxis: {
                    title: '# of Imports',
                    tickfont: {color: '#a1a1aa', size: 10},
                    gridcolor: '#374151',
                    color: '#a1a1aa'
                },
                yaxis: {
                    title: '',
                    tickfont: {color: '#a1a1aa', size: 10},
                    color: '#a1a1aa',
                    autorange: 'reversed'
                },
                legend: {
                    orientation: 'h',
                    x: 0.5,
                    xanchor: 'center',
                    y: 1.15,
                    yanchor: 'bottom',
                    font: {color: '#a1a1aa', size: 9}
                }
            };

            Plotly.newPlot('country-chart', [traceTP, traceTN, tracePN], layout, {displayModeBar: false});

            const countryChart = document.getElementById('country-chart');
            countryChart.removeAllListeners && countryChart.removeAllListeners('plotly_click');

            countryChart.on('plotly_click', function(data) {
                const country = data.points[0].y;  // Changed from x to y for horizontal bars
                if (dimensions.CTRY_CODE.hasCurrentFilter()) {
                    dimensions.CTRY_CODE.filterAll();
                } else {
                    dimensions.CTRY_CODE.filter(country);
                }
                updateAll();
            });
        }
    };
}

function createSankeyChart() {
    return {
        update: function() {
            const allRecords = ndx.allFiltered();

            // Build flow data: TP_FLAG -> MONTH -> CTRY_CODE
            const flowCounts = {};

            allRecords.forEach(d => {
                if (d.TP_FLAG && d.MONTH && d.CTRY_CODE) {
                    const key = `${d.TP_FLAG}|${d.MONTH}|${d.CTRY_CODE}`;
                    flowCounts[key] = (flowCounts[key] || 0) + 1;
                }
            });

            // Create unique node labels and indices
            const nodeLabels = new Set();
            const tpFlags = new Set();
            const months = new Set();
            const countries = new Set();

            Object.keys(flowCounts).forEach(key => {
                const [tp, month, country] = key.split('|');
                tpFlags.add(tp);
                months.add(month);
                countries.add(country);
            });

            // Create node arrays in order: TP_FLAG nodes, MONTH nodes, CTRY_CODE nodes
            const tpFlagNodes = Array.from(tpFlags).sort();
            const monthNodes = Array.from(months).sort((a, b) => {
                const monthOrder = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                return monthOrder.indexOf(a) - monthOrder.indexOf(b);
            });
            const countryNodes = Array.from(countries).sort();

            // Combine all nodes
            const allNodes = [...tpFlagNodes, ...monthNodes, ...countryNodes];

            // Create node index map
            const nodeIndex = {};
            allNodes.forEach((node, i) => {
                nodeIndex[node] = i;
            });

            // Create links for TP_FLAG -> MONTH
            const links = {
                source: [],
                target: [],
                value: [],
                color: []
            };

            const tpMonthFlows = {};
            const monthCountryFlows = {};

            Object.entries(flowCounts).forEach(([key, count]) => {
                const [tp, month, country] = key.split('|');

                // Aggregate TP_FLAG -> MONTH
                const tpMonthKey = `${tp}|${month}`;
                tpMonthFlows[tpMonthKey] = (tpMonthFlows[tpMonthKey] || 0) + count;

                // Aggregate MONTH -> CTRY_CODE
                const monthCountryKey = `${month}|${country}`;
                monthCountryFlows[monthCountryKey] = (monthCountryFlows[monthCountryKey] || 0) + count;
            });

            // Color mapping for TP_FLAG
            const tpColors = {
                'TP': 'rgba(244, 67, 54, 0.3)',
                'TN': 'rgba(76, 175, 80, 0.3)',
                'PN': 'rgba(255, 152, 0, 0.3)'
            };

            // Add TP_FLAG -> MONTH links
            Object.entries(tpMonthFlows).forEach(([key, count]) => {
                const [tp, month] = key.split('|');
                links.source.push(nodeIndex[tp]);
                links.target.push(nodeIndex[month]);
                links.value.push(count);
                links.color.push(tpColors[tp] || 'rgba(128, 128, 128, 0.3)');
            });

            // Add MONTH -> CTRY_CODE links
            Object.entries(monthCountryFlows).forEach(([key, count]) => {
                const [month, country] = key.split('|');
                links.source.push(nodeIndex[month]);
                links.target.push(nodeIndex[country]);
                links.value.push(count);
                links.color.push('rgba(100, 100, 100, 0.2)');
            });

            // Create Sankey trace
            const data = [{
                type: 'sankey',
                orientation: 'h',
                node: {
                    pad: 15,
                    thickness: 20,
                    line: {
                        color: '#374151',
                        width: 1
                    },
                    label: allNodes,
                    color: allNodes.map(node => {
                        if (tpFlagNodes.includes(node)) {
                            return node === 'TP' ? '#F44336' : node === 'TN' ? '#4CAF50' : '#FFEB3B';
                        } else if (monthNodes.includes(node)) {
                            return '#3B82F6';
                        } else {
                            return '#8B5CF6';
                        }
                    }),
                    customdata: allNodes.map(node => {
                        if (tpFlagNodes.includes(node)) return 'TP_FLAG';
                        if (monthNodes.includes(node)) return 'MONTH';
                        return 'CTRY_CODE';
                    }),
                    hovertemplate: '%{label}<br>%{customdata}<br>Total: %{value}<extra></extra>'
                },
                link: links
            }];

            const layout = {
                height: 300,
                margin: {t: 10, r: 10, b: 10, l: 10},
                paper_bgcolor: '#1a1a1a',
                plot_bgcolor: '#1a1a1a',
                font: {
                    color: '#a1a1aa',
                    size: 10
                }
            };

            Plotly.newPlot('sankey-chart', data, layout, {displayModeBar: false});
        }
    };
}

function updateDataTable() {
    const filteredData = Object.values(dimensions)[0]?.top(Infinity) || [];

    // Specify the columns we want to display
    const priorityColumns = ['TP_FLAG', 'MONTH', 'CTRY_CODE', 'CTRY_MONTH', 'ENTY_ID'];
    const displayColumns = priorityColumns.filter(col =>
        categoricalColumns.includes(col) || numericColumns.includes(col)
    );

    // Prepare table data for Plotly
    const headerValues = ['#', ...displayColumns];
    const cellValues = [
        // Row numbers
        filteredData.map((d, i) => i + 1),
        // Data columns
        ...displayColumns.map(col =>
            filteredData.map(d => {
                const value = d[col];
                return typeof value === 'number' ? value.toFixed(3) : value;
            })
        )
    ];

    const tableData = [{
        type: 'table',
        columnwidth: [40, ...displayColumns.map(() => 80)], // Smaller column widths
        header: {
            values: headerValues,
            align: 'center',
            line: {width: 1, color: '#374151'},
            fill: {color: '#242424'},
            font: {family: "Arial", size: 10, color: "white"}
        },
        cells: {
            values: cellValues,
            align: 'center',
            line: {width: 1, color: '#374151'},
            fill: {color: [
                filteredData.map((_, i) => i % 2 === 0 ? '#1a1a1a' : '#2a2a2a')
            ]},
            font: {family: "Arial", size: 9, color: "#a1a1aa"}
        }
    }];

    const layout = {
        margin: {t: 10, b: 10, l: 10, r: 10},
        paper_bgcolor: '#1a1a1a',
        plot_bgcolor: '#1a1a1a',
        height: 620 // Adjusted to fit within container
    };

    Plotly.newPlot('plotly-table', tableData, layout, {displayModeBar: false});
}

function updateStats() {
    const filteredCount = all.value();
    const totalCount = originalData.length;
    
    d3.select('#filtered-count').text(filteredCount.toLocaleString());
    d3.select('#total-count').text(totalCount.toLocaleString());
}

function updateAll() {
    // Only update charts that exist on current dashboard
    if (currentDashboard === 'dashboard1') {
        Object.values(charts).forEach(chart => {
            if (chart.update) chart.update();
        });
        if (typeof updateDataTable === 'function') updateDataTable();
        if (typeof updateStats === 'function') updateStats();
    } else if (currentDashboard === 'dashboard2') {
        // Update Dashboard 2 charts
        if (typeof createSunburstChart === 'function') {
            createSunburstChart();
        }
        if (typeof updateDataTableDashboard2 === 'function') {
            updateDataTableDashboard2();
        }
    } else if (currentDashboard === 'dashboard3') {
        // Update Dashboard 3 charts (placeholder for future implementation)
        console.log('Dashboard 3 update - ready for custom charts');
    }
}

// Global reset functions
window.resetAllFilters = function() {
    Object.values(dimensions).forEach(dim => dim.filterAll());
    updateAll();
}

window.resetTargetFilter = function() {
    if (dimensions.TP_FLAG) {
        dimensions.TP_FLAG.filterAll();
        updateAll();
    }
}

window.resetMonthFilter = function() {
    if (dimensions.MONTH) {
        dimensions.MONTH.filterAll();
        updateAll();
    }
}

window.resetCountryFilter = function() {
    if (dimensions.CTRY_CODE) {
        dimensions.CTRY_CODE.filterAll();
        updateAll();
    }
}

window.resetFilter = function(column) {
    if (dimensions[column]) {
        dimensions[column].filterAll();
        updateAll();
    }
}

// Initialize on page load
initDashboard();