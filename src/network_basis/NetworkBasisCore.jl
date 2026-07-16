struct NetworkBasisCore{NT, BT, SNNT, QWFT, VT, VWFT}
    activation
    NN        :: NT
    backend   :: BT
    SNN       :: SNNT
    dqdθ      :: QWFT
    V_func    :: VT
    dvdθ      :: VWFT
end
