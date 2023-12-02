<template>
    <div class="chart" style="width: 100%; height: 280px;"></div>
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
      let counts = data.genres.map(item => item.scoreGT5Count)
      let max = Math.max(...counts)
      data.genres.forEach(item => {
        newData.push({
          name: item.genre,
          value: item.scoreGT5Count,
          max: max
        })
      })
      let option = {
        title: {
          text: 'Movies Top',
          left: 10
        },
        radar: {
          indicator: newData,
          radius: '70%',
          center: ['50%', '55%']
        },
        series: [
          {
            name: 'Budget vs spending',
            type: 'radar',
            data: [
              {
                value: counts,
              }
            ]
          },
        ]
      };
      this.chart.setOption(option);
    },
  },
}
</script>