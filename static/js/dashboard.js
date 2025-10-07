// Get filename from URL parameter
const urlParams = new URLSearchParams(window.location.search);
const filename = urlParams.get('file') || 'sample_features.csv';

// Global variables
let ndx, all;
let dimensions = {};
let groups = {};
let charts = {};
let numericColumns = [];
let categoricalColumns = [];
let originalData = [];

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
        const response = await fetch(`/api/csv-data?filename=${encodeURIComponent(filename)}`);
        
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
            <div class="chart-container" style="width: 250px;">
                <div class="chart-title">
                    Target by Category
                    <button class="reset-btn-small" onclick="resetTargetFilter()">Reset</button>
                </div>
                <svg id="pie-chart" width="210" height="190"></svg>
            </div>

            <div class="chart-container" style="flex: 1;">
                <div class="chart-title">
                    Monthly Import
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
    `;

    // Add data table (Plotly container)
    html += `
        <div class="chart-container">
            <div class="chart-title">📋 Filtered Data</div>
            <div id="plotly-table" style="width: 100%;"></div>
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
}

function createCategoryPieChart(tooltip) {
    const width = 210;  // 250px container - 40px padding (20px each side)
    const height = 200; // Maximized height to fill available space
    const radius = Math.min(width, height) / 2 - 5;

    const svg = d3.select('#pie-chart');
    const g = svg.append('g')
        .attr('transform', `translate(${width/2},${height/2})`);

    const pie = d3.pie()
        .value(d => d.value)
        .sort(null);

    const path = d3.arc()
        .outerRadius(radius - 10)
        .innerRadius(radius * 0.3);

    const labelArc = d3.arc()
        .outerRadius(radius - 30)
        .innerRadius(radius - 30);

    // Fixed colors for TARGET_PROXY categories
    const categoryColors = d3.scaleOrdinal()
        .domain(['TP', 'TN', 'PN'])
        .range(['#F44336','#4CAF50','#FF9800']);

    // Create TARGET_PROXY dimension if it doesn't exist
    if (!dimensions.TARGET_PROXY) {
        dimensions.TARGET_PROXY = ndx.dimension(d => d.TARGET_PROXY || 'Unknown');
        groups.TARGET_PROXY = dimensions.TARGET_PROXY.group().reduceCount();
    }
    
    return {
        update: function() {
            // Get TARGET_PROXY counts from crossfilter
            const data = groups.TARGET_PROXY.all().filter(d => d.value > 0);
            console.log('TARGET_PROXY data from crossfilter:', data);

            this.renderPieChart(data);
        },

        renderPieChart: function(data) {
            console.log('Rendering pie chart with data:', data);

            if (data.length === 0) return;

            const arcs = g.selectAll('.arc')
                .data(pie(data));

            const arcEnter = arcs.enter().append('g')
                .attr('class', 'arc');

            arcEnter.append('path')
                .attr('class', 'pie-slice');

            arcEnter.append('text')
                .attr('class', 'pie-label');

            arcs.exit().remove();

            const arcUpdate = arcEnter.merge(arcs);

            arcUpdate.select('path')
                .attr('d', path)
                .attr('fill', d => categoryColors(d.data.key))
                .attr('stroke', 'white')
                .attr('stroke-width', 2)
                .on('mouseover', function(event, d) {
                    tooltip.style('display', 'block')
                        .html(`${d.data.key}<br/>Count: ${d.data.value.toLocaleString()}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                })
                .on('mouseout', function() {
                    tooltip.style('display', 'none');
                })
                .on('click', function(event, d) {
                    if (dimensions.TARGET_PROXY.hasCurrentFilter()) {
                        dimensions.TARGET_PROXY.filterAll();
                    } else {
                        dimensions.TARGET_PROXY.filter(d.data.key);
                    }
                    updateAll();
                });

            arcUpdate.select('text')
                .attr('transform', d => `translate(${labelArc.centroid(d)})`)
                .attr('dy', '0.35em')
                .style('text-anchor', 'middle')
                .style('font-size', '11px')
                .style('fill', 'white')
                .style('font-weight', 'bold')
                .text(d => d.data.key);
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

            // Count all TARGET_PROXY types per month
            const monthCounts = {};
            allRecords.forEach(record => {
                const month = record.MONTH;
                if (month) {
                    if (!monthCounts[month]) {
                        monthCounts[month] = {TP: 0, TN: 0, PN: 0};
                    }
                    const target = record.TARGET_PROXY;
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
                marker: {color: '#FF9800'},
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

            // Count all TARGET_PROXY types per country
            const countryCounts = {};
            allRecords.forEach(d => {
                if (d.CTRY_CODE && d.TARGET_PROXY) {
                    if (!countryCounts[d.CTRY_CODE]) {
                        countryCounts[d.CTRY_CODE] = {TP: 0, TN: 0, PN: 0, total: 0};
                    }
                    const target = d.TARGET_PROXY;
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
                marker: {color: '#FF9800'},
                hovertemplate: '%{y}<br>PN: %{x}<extra></extra>'
            };

            const layout = {
                width: 300,
                height: 240,
                barmode: 'stack',
                margin: {t: 30, r: 20, b: 40, l: 30},
                paper_bgcolor: '#1a1a1a',
                plot_bgcolor: '#242424',
                xaxis: {
                    title: 'Count',
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

function updateDataTable() {
    const filteredData = Object.values(dimensions)[0]?.top(Infinity) || [];

    // Specify the columns we want to display
    const priorityColumns = ['TARGET_PROXY', 'MONTH', 'CTRY_CODE', 'CTRY_MONTH', 'ENTY_ID'];
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
        header: {
            values: headerValues,
            align: 'center',
            line: {width: 1, color: '#374151'},
            fill: {color: '#242424'},
            font: {family: "Arial", size: 12, color: "white"}
        },
        cells: {
            values: cellValues,
            align: 'center',
            line: {width: 1, color: '#374151'},
            fill: {color: ['#1a1a1a', 'rgba(255, 255, 255, 0.02)']},
            font: {family: "Arial", size: 11, color: "#a1a1aa"}
        }
    }];

    const layout = {
        margin: {t: 10, b: 10, l: 10, r: 10},
        paper_bgcolor: '#1a1a1a',
        plot_bgcolor: '#1a1a1a',
        height: 300
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
    Object.values(charts).forEach(chart => {
        if (chart.update) chart.update();
    });
    updateDataTable();
    updateStats();
}

// Global reset functions
window.resetAllFilters = function() {
    Object.values(dimensions).forEach(dim => dim.filterAll());
    updateAll();
}

window.resetTargetFilter = function() {
    if (dimensions.TARGET_PROXY) {
        dimensions.TARGET_PROXY.filterAll();
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