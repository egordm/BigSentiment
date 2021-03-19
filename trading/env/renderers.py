import os
from collections import OrderedDict
from typing import Union, Tuple

import pandas as pd
import plotly.graph_objects as go
from IPython.display import display, clear_output
from plotly.subplots import make_subplots
from tensortrade.env.default.renderers import BaseRenderer, _check_valid_format, _check_path, _create_auto_file_name


class PlotlyCustomChart(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 height: int = None,
                 timestamp_format: str = '%Y-%m-%d %H:%M:%S',
                 save_format: str = None,
                 path: str = 'charts',
                 filename_prefix: str = 'chart_',
                 auto_open_html: bool = False,
                 include_plotlyjs: Union[bool, str] = 'cdn') -> None:
        super().__init__()
        self._height = height
        self._timestamp_format = timestamp_format
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self._include_plotlyjs = include_plotlyjs
        self._auto_open_html = auto_open_html

        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

        self.fig = None
        self._price_chart = None
        self._volume_chart = None
        self._performance_chart = None
        self._net_worth_chart = None
        self._base_annotations = None
        self._last_trade_step = 0
        self._show_chart = display

    def _create_figure(self, performance_keys: dict) -> None:
        # Make whole froma
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.15, 0.15],
        )
        # Add price ticker
        fig.add_trace(go.Candlestick(name='Price', xaxis='x1', yaxis='y1', showlegend=False), row=1, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Add volume line
        fig.add_trace(go.Bar(name='Volume', showlegend=False, marker={'color': 'DodgerBlue'}), row=2, col=1)
        for k in performance_keys:
            fig.add_trace(go.Scatter(mode='lines', name=k), row=3, col=1)
        fig.add_trace(go.Scatter(mode='lines', name='Net Worth', marker={'color': 'DarkGreen'}), row=4, col=1)

        fig.update_xaxes(linecolor='Grey', gridcolor='Gainsboro')
        fig.update_yaxes(linecolor='Grey', gridcolor='Gainsboro')
        fig.update_xaxes(title_text='Price', row=1)
        fig.update_xaxes(title_text='Volume', row=2)
        fig.update_xaxes(title_text='Performance', row=3)
        fig.update_xaxes(title_text='Net Worth', row=4)
        fig.update_xaxes(title_standoff=7, title_font=dict(size=12))

        self.fig = go.FigureWidget(fig)
        self._price_chart = self.fig.data[0]
        self._volume_chart = self.fig.data[1]
        self._performance_chart = self.fig.data[2]
        self._net_worth_chart = self.fig.data[-1]

        self.fig.update_annotations({'font': {'size': 12}})
        self.fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
        self._base_annotations = self.fig.layout.annotations

    def _create_trade_annotations(
            self, trades: 'OrderedDict', price_history: 'pd.DataFrame'
    ) -> 'Tuple[go.layout.Annotation]':
        annotations = []
        for trade in reversed(trades.values()):
            trade = trade[0]

            tp = float(trade.price)
            ts = float(trade.size)

            if trade.step <= self._last_trade_step:
                break

            if trade.side.value == 'buy':
                color = 'DarkGreen'
                ay = 15
                qty = round(ts / tp, trade.quote_instrument.precision)

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'],
                    side=trade.side.value.upper(),
                    qty=qty,
                    size=ts,
                    quote_instrument=trade.quote_instrument,
                    price=tp,
                    base_instrument=trade.base_instrument,
                    type=trade.type.value.upper(),
                    commission=trade.commission
                )

            elif trade.side.value == 'sell':
                color = 'FireBrick'
                ay = -15
                # qty = round(ts * tp, trade.quote_instrument.precision)

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'],
                    side=trade.side.value.upper(),
                    qty=ts,
                    size=round(ts * tp, trade.base_instrument.precision),
                    quote_instrument=trade.quote_instrument,
                    price=tp,
                    base_instrument=trade.base_instrument,
                    type=trade.type.value.upper(),
                    commission=trade.commission
                )
            else:
                raise ValueError(f"Valid trade side values are 'buy' and 'sell'. Found '{trade.side.value}'.")

            hovertext = 'Step {step} [{datetime}]<br>' \
                        '{side} {qty} {quote_instrument} @ {price} {base_instrument} {type}<br>' \
                        'Total: {size} {base_instrument} - Comm.: {commission}'.format(**text_info)

            annotations += [go.layout.Annotation(
                x=trade.step - 1, y=tp,
                ax=15,
                ay=ay * 2.2,
                align="center", opacity=0.6,
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=3.5, arrowcolor=color,
                hovertext=hovertext, hoverlabel=dict(bgcolor=color),
            )]

        if trades:
            self._last_trade_step = trades[list(trades)[-1]][0].step

        return tuple(annotations)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        if price_history is None:
            raise ValueError("renderers() is missing required positional argument 'price_history'.")

        if net_worth is None:
            raise ValueError("renderers() is missing required positional argument 'net_worth'.")

        if performance is None:
            raise ValueError("renderers() is missing required positional argument 'performance'.")

        if trades is None:
            raise ValueError("renderers() is missing required positional argument 'trades'.")

        if not self.fig:
            self._create_figure(performance.keys())

        if self._show_chart:  # ensure chart visibility through notebook cell reruns
            display(self.fig)

        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self._price_chart.update(dict(
            open=price_history['open'],
            high=price_history['high'],
            low=price_history['low'],
            close=price_history['close']
        ))
        self.fig.layout.annotations += self._create_trade_annotations(trades, price_history)

        self._volume_chart.update({'y': price_history['volume']})

        for trace in self.fig.select_traces(row=3):
            trace.update({'y': performance[trace.name]})

        self._net_worth_chart.update({'y': net_worth})

        if self._show_chart:
            self.fig.show()

    def save(self) -> None:
        """Saves the current chart to a file.

        Notes
        -----
        All formats other than HTML require Orca installed and server running.
        """
        if not self._save_format:
            return
        else:
            valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
            _check_valid_format(valid_formats, self._save_format)

        _check_path(self._path)

        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=self._auto_open_html)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self._last_trade_step = 0
        if self.fig is None:
            return

        self.fig.layout.annotations = self._base_annotations
        clear_output(wait=True)
