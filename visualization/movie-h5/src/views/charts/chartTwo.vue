<template>
    <div class="chart" style="width: 100%; height: 320px;"></div>
</template>
<script>
import * as echarts from 'echarts'
require('echarts/theme/macarons') // echarts theme
import resize from './mixins/resize'
export default {
  mixins: [resize],
  data() {
    return {
      chart: null,
    }
  },
  mounted() {
    this.$nextTick(() => {
      this.chart = echarts.init(
        this.$el,
        "macarons"
      );
      this.initChart();
    });
  },
  beforeDestroy() {
    if (!this.chart) {
      return
    }
    this.chart.dispose()
    this.chart = null
  },
  methods: {
    initChart() {
      let data = require('@/utils/simple.json')
      console.log(data)
      let newData = []
      data.genres.forEach(item => {
        newData.push({
          name: item.genre,
          value: item.scoreGT5Count,
        })
      })
      let afterPx = newData.sort((a, b) => {return a.value - b.value})
      let option = {
        grid: [
          {
			top: 0,
			bottom: '8%',
            left: '30%',
            right: '5%'
          }
        ],
        xAxis: [
          {
            type: 'value'
          }
        ],
        yAxis: [
          {
            type: 'category',
            data: afterPx.map(item => item.name)
          }
        ],
        series: [
          {
            type: 'bar',
            data: afterPx
          }
        ]
      };
      this.chart.setOption(option);
    },
  },
}
</script>