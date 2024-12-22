$(document).ready(function () {
    const ctx = document.getElementById('commentsChart').getContext('2d');

    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Comments Per Hour',
                data: [],
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    enabled: true,
                    callbacks: {
                        label: function (context) {
                            return `${context.label}: ${context.raw} comments`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Comments Per Hour'
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Date and Hour' },
                    ticks: { autoSkip: true, maxTicksLimit: 10 }
                },
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Number of Comments' }
                }
            }
        }
    });

    let selectedSubreddit = 'movies';

    function updateInfoCards(data) {
        $('#total-comments').text(data.total_comments || 0);
        $('#peak-hour').text(data.peak_hour || '0:00');
        $('#peak-comments').text(data.peak_comments || 0);
        $('#total-deleted-posts').text(data.total_deleted_posts || 0);
        $('#active-users').text(data.active_users || 0);
        $('#avg-comments-hour').text(data.avg_comments_hour || 0);
    }

    function updateData() {
        const startDate = $('#startDate').val();
        const endDate = $('#endDate').val();

        $.getJSON(`/get_comments_data?subreddit=${selectedSubreddit}&start_date=${startDate}&end_date=${endDate}`, function (data) {
            chart.data.labels = data.labels;
            chart.data.datasets[0].data = data.data;
            chart.update();

            updateInfoCards(data);

            const peakValue = Math.max(...data.data);
            const peakIndex = data.data.indexOf(peakValue);

            if (peakIndex !== -1) {
                const meta = chart.getDatasetMeta(0);
                const dataPoint = meta.data[peakIndex];

                if (dataPoint) {
                    const x = dataPoint.x;
                    const y = dataPoint.y;

                    $('#peakMarker').css({
                        left: `${x}px`,
                        top: `${y}px`,
                        transform: 'translate(-50%, -50%)',
                        display: 'block'
                    });
                }
            }
        }).fail(function (jqXHR, textStatus, errorThrown) {
            console.error(`Failed to fetch comments data for r/${selectedSubreddit}:`, errorThrown);
            alert("Unable to load data. Please check the server or your connection.");
        });

        chart.options.plugins.title.text = `Comments Per Hour (r/${selectedSubreddit})`;
        chart.update();
    }

    $('.dropdown-item').on('click', function () {
        selectedSubreddit = $(this).data('subreddit');
        updateData();
    });

    $('#updateCommentsChart').on('click', function () {
        updateData();
    });

    updateData();
});