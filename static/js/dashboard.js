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
                processed[key] = String(value || '');
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
        <div class="dashboard">
            <div class="chart-container left-chart">
                <div class="chart-title">
                    Target by Category
                    <button class="reset-btn-small" onclick="resetTargetFilter()">Reset</button>
                </div>
                <svg id="pie-chart" width="210" height="190"></svg>
            </div>

            <div class="chart-container right-chart">
                <div class="chart-title">
                    Monthly Import
                    <button class="reset-btn-small" onclick="resetMonthFilter()">Reset</button>
                </div>
                <svg id="bar-chart" width="460" height="200"></svg>
            </div>
        </div>
    `;
    
    // Add data table
    html += `
        <div class="chart-container">
            <div class="chart-title">📋 Filtered Data</div>
            <div id="data-table-container">
                <table class="data-table" id="data-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            ${[...numericColumns.slice(0, 3), ...categoricalColumns.slice(0, 2)]
                                .map(col => `<th>${col}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody id="table-body"></tbody>
                </table>
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
}

function createCategoryPieChart(tooltip) {
    const width = 210;  // 250px container - 40px padding (20px each side)
    const height = 190; // Maximized height to fill available space
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
    const width = 460;
    const height = 200;
    const margin = {top: 10, right: 20, bottom: 30, left: 40};

    const svg = d3.select('#bar-chart');
    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const x = d3.scaleBand().range([0, innerWidth]).padding(0.1);
    const y = d3.scaleLinear().range([innerHeight, 0]);

    const xAxis = g.append('g')
        .attr('class', 'axis axis--x')
        .attr('transform', `translate(0,${innerHeight})`);

    const yAxis = g.append('g')
        .attr('class', 'axis axis--y');

    // Color mapping
    const targetColors = {
        'TP': '#F44336',
        'TN': '#4CAF50',
        'PN': '#FF9800'
    };

    // Create MONTH dimension if it doesn't exist
    if (!dimensions.MONTH) {
        dimensions.MONTH = ndx.dimension(d => {
            if (d.MONTH) {
                const monthNum = parseInt(d.MONTH.replace('x', ''));
                if (!isNaN(monthNum)) {
                    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                    return months[monthNum - 1] || d.MONTH;
                }
            }
            return 'Unknown';
        });
        groups.MONTH = dimensions.MONTH.group().reduceCount();
    }

    return {
        update: function() {
            let data = groups.MONTH.all().filter(d => d.value > 0);

            const monthOrder = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            data.sort((a, b) => monthOrder.indexOf(a.key) - monthOrder.indexOf(b.key));

            // Build map of each row's TARGET_PROXY for coloring
            const allRecords = dimensions.MONTH.top(Infinity);
            const monthDominantTarget = {};

            allRecords.forEach(record => {
                const monthNum = parseInt(record.MONTH?.replace('x', ''));
                if (!isNaN(monthNum)) {
                    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                    const month = months[monthNum - 1];

                    if (!monthDominantTarget[month]) {
                        monthDominantTarget[month] = {TP: 0, TN: 0, PN: 0};
                    }

                    const target = record.TARGET_PROXY;
                    if (target && (target === 'TP' || target === 'TN' || target === 'PN')) {
                        monthDominantTarget[month][target]++;
                    }
                }
            });

            // Attach dominant target to each bar
            data.forEach(d => {
                const counts = monthDominantTarget[d.key] || {TP: 0, TN: 0, PN: 0};
                let maxTarget = 'TP';
                let maxCount = counts.TP;

                if (counts.TN > maxCount) {
                    maxTarget = 'TN';
                    maxCount = counts.TN;
                }
                if (counts.PN > maxCount) {
                    maxTarget = 'PN';
                }

                d.dominantTarget = maxTarget;
            });

            x.domain(data.map(d => d.key));
            y.domain([0, d3.max(data, d => d.value)]);

            xAxis.call(d3.axisBottom(x));
            yAxis.call(d3.axisLeft(y));

            const bars = g.selectAll('.bar').data(data);

            bars.enter()
                .append('rect')
                .attr('class', 'bar')
                .merge(bars)
                .attr('x', d => x(d.key))
                .attr('y', d => y(d.value))
                .attr('width', x.bandwidth())
                .attr('height', d => innerHeight - y(d.value))
                .attr('fill', d => targetColors[d.dominantTarget])
                .on('mouseover', function(event, d) {
                    tooltip.style('display', 'block')
                        .html(`${d.key}<br/>Imports: ${d.value}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                })
                .on('mouseout', function() {
                    tooltip.style('display', 'none');
                })
                .on('click', function(event, d) {
                    if (dimensions.MONTH.hasCurrentFilter()) {
                        dimensions.MONTH.filterAll();
                    } else {
                        dimensions.MONTH.filter(d.key);
                    }
                    updateAll();
                });

            bars.exit().remove();
        }
    };
}

function updateDataTable() {
    const filteredData = Object.values(dimensions)[0]?.top(Infinity) || [];
    const displayColumns = [...numericColumns.slice(0, 3), ...categoricalColumns.slice(0, 2)];

    const tbody = d3.select('#table-body');
    const rows = tbody.selectAll('tr').data(filteredData);

    const rowEnter = rows.enter().append('tr');
    // Add row number cell plus data cells
    rowEnter.append('td'); // for row number
    displayColumns.forEach(() => rowEnter.append('td'));

    rows.exit().remove();

    const rowUpdate = rowEnter.merge(rows);

    // Update row number (first column)
    rowUpdate.select('td:nth-child(1)')
        .text((d, i) => i + 1);

    // Update data columns
    displayColumns.forEach((col, i) => {
        rowUpdate.select(`td:nth-child(${i + 2})`)
            .text(d => {
                const value = d[col];
                return typeof value === 'number' ? value.toFixed(3) : value;
            });
    });
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

window.resetFilter = function(column) {
    if (dimensions[column]) {
        dimensions[column].filterAll();
        updateAll();
    }
}

// Initialize on page load
initDashboard();