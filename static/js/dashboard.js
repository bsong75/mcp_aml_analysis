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
                        <svg id="pie-chart" width="210" height="190"></svg>
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
    const width = 210;  // 250px container - 40px padding (20px each side)
    const height = 200; // Maximized height to fill available space
    const radius = Math.min(width, height) / 2 - 1;

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
        .range(['#F44336','#4CAF50','#FFEB3B']);

    // Create TP_FLAG dimension if it doesn't exist
    if (!dimensions.TP_FLAG) {
        dimensions.TP_FLAG = ndx.dimension(d => d.TP_FLAG || 'Unknown');
        groups.TP_FLAG = dimensions.TP_FLAG.group().reduceCount();
    }
    
    return {
        update: function() {
            // Get TP_FLAG counts from crossfilter
            const data = groups.TP_FLAG.all().filter(d => d.value > 0);
            console.log('TP_FLAG data from crossfilter:', data);

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
                    if (dimensions.TP_FLAG.hasCurrentFilter()) {
                        dimensions.TP_FLAG.filterAll();
                    } else {
                        dimensions.TP_FLAG.filter(d.data.key);
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